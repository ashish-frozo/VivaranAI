"""
Database utility functions for chat context and bill analysis persistence.
"""

import uuid
import logging
import sqlalchemy as sa
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from database.repositories import BillAnalysisRepository
from database.models import BillAnalysis

# In-memory fallback storage for bill analyses
_in_memory_bills = {}

logger = logging.getLogger(__name__)

async def ensure_user_exists(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """
    Ensure a user exists in the database with the given ID.
    If not, create a default user record.
    
    Args:
        session: Database session
        user_id: UUID of the user to check/create
        
    Returns:
        bool: True if user exists or was created, False if operation failed
    """
    try:
        # Check if user exists
        query = sa.text("SELECT id FROM users WHERE id = :user_id")
        result = await session.execute(query, {"user_id": user_id})
        user_exists = result.scalar() is not None
        
        if not user_exists:
            # Create a default user
            logger.info(f"Creating default user with id={user_id}")
            insert_query = sa.text("""
                INSERT INTO users (id, email, name, created_at, updated_at) 
                VALUES (:user_id, :email, :name, NOW(), NOW())
            """)
            await session.execute(
                insert_query, 
                {
                    "user_id": user_id,
                    "email": f"default_{str(user_id)[:8]}@example.com",
                    "name": f"Default User {str(user_id)[:8]}"
                }
            )
            await session.commit()
            logger.info(f"Created default user with id={user_id}")
            return True
        
        return True
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to ensure user exists: {e}")
        return False


def serialize_for_json(obj):
    """Recursively serialize objects for JSON storage, handling complex Python objects"""
    import types
    from enum import Enum
    
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, types.MappingProxyType):
        # Handle mappingproxy objects (like enum.__members__)
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, Enum):
        # Handle enum values
        return obj.value
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        # Handle custom objects (but not classes themselves)
        try:
            return serialize_for_json(obj.__dict__)
        except (TypeError, AttributeError):
            return str(obj)
    elif callable(obj):
        # Handle functions/methods
        return str(obj)
    else:
        # Handle primitive types and fallback to string for complex objects
        try:
            # Test if object is JSON serializable
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


async def save_bill_analysis(session: AsyncSession, user_id: str, doc_id: str, filename: str, file_hash: str, file_size: int, content_type: str, analysis_type: str, raw_analysis: Dict[str, Any], structured_results: Dict[str, Any], status: str = "completed") -> BillAnalysis:
    # Handle non-UUID format strings by generating new UUIDs if needed
    try:
        id_uuid = uuid.UUID(doc_id)
    except ValueError:
        # If doc_id is not a valid UUID format, generate a new one based on the string
        id_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
    
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        # If user_id is not a valid UUID format, generate a new one based on the string
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
    
    # Convert UUIDs to strings for consistent dictionary keys
    id_str = str(id_uuid)
    user_id_str = str(user_uuid)
    
    # Log the UUIDs being used
    logger.info(f"save_bill_analysis: Using doc_id={id_str}, user_id={user_id_str}")
    
    try:
        # Serialize datetime objects and other non-JSON-serializable objects
        serialized_raw_analysis = serialize_for_json(raw_analysis)
        serialized_structured_results = serialize_for_json(structured_results)
        
        # Try to use the database first with robust session management
        try:
            repo = BillAnalysisRepository(session)
            analysis_data = {
                "id": id_uuid,
                "user_id": user_uuid,
                "filename": filename,
                "file_hash": file_hash,
                "file_size": file_size,
                "content_type": content_type,
                "analysis_type": analysis_type,
                "status": status,
                "raw_analysis": serialized_raw_analysis,
                "structured_results": serialized_structured_results,
            }
            
            try:
                # Create analysis with explicit transaction handling
                result = await repo.create_analysis(analysis_data)
                await session.commit()
                logger.info(f"save_bill_analysis: Successfully saved bill analysis to database for doc_id={id_str}")
                return result
            except Exception as fk_error:
                # Check if it's a foreign key violation for user_id
                if "violates foreign key constraint" in str(fk_error) and "bill_analyses_user_id_fkey" in str(fk_error):
                    # Rollback the failed transaction
                    await session.rollback()
                    logger.warning(f"Foreign key violation for user_id={user_id_str}, attempting to create default user")
                    
                    # Try to create a default user
                    user_created = await ensure_user_exists(session, user_uuid)
                    if user_created:
                        # Try again with the user created
                        logger.info(f"Retrying bill analysis creation after creating default user")
                        result = await repo.create_analysis(analysis_data)
                        await session.commit()
                        logger.info(f"Successfully saved bill analysis after creating default user")
                        return result
                    else:
                        # If user creation failed, fall back to in-memory storage
                        logger.warning(f"Failed to create default user, using in-memory fallback")
                        raise Exception(f"Could not create default user: {user_id_str}")
                else:
                    # Rollback on any other database error
                    await session.rollback()
                    raise fk_error
        except Exception as db_error:
            # Log the specific error
            logger.error(f"Database error in save_bill_analysis: {db_error}")
            raise db_error
    except Exception as e:
        # Fallback to in-memory storage if database fails
        logger.warning(f"Database operation failed, using in-memory fallback: {e}")
        
        # Initialize user's bills list if not exists
        if user_id_str not in _in_memory_bills:
            _in_memory_bills[user_id_str] = {}
        
        # Create a timestamp for created_at
        import datetime
        now = datetime.datetime.now().isoformat()
        
        # Store the bill analysis in memory
        bill_data = {
            "id": id_str,
            "user_id": user_id_str,
            "filename": filename,
            "file_hash": file_hash,
            "file_size": file_size,
            "content_type": content_type,
            "analysis_type": analysis_type,
            "status": status,
            "raw_analysis": raw_analysis,
            "structured_results": structured_results,
            "created_at": now
        }
        
        # Debug log the raw_analysis and structured_results
        logger.debug(f"Saving bill analysis for doc_id={id_str}")
        logger.debug(f"raw_analysis type: {type(raw_analysis)}")
        logger.debug(f"structured_results type: {type(structured_results)}")
        
        if isinstance(raw_analysis, dict):
            logger.debug(f"raw_analysis keys: {list(raw_analysis.keys())}")
            if 'final_result' in raw_analysis:
                logger.debug(f"final_result keys: {list(raw_analysis['final_result'].keys())}")
        
        if isinstance(structured_results, dict):
            logger.debug(f"structured_results keys: {list(structured_results.keys())}")
        
        _in_memory_bills[user_id_str][id_str] = bill_data
        
        # Create a mock BillAnalysis object
        mock_analysis = type('BillAnalysis', (), bill_data)
        return mock_analysis

async def get_user_bills(session: AsyncSession, user_id: str, limit: int = 50) -> List[BillAnalysis]:
    # Handle non-UUID format strings by generating a UUID if needed
    try:
        user_uuid = uuid.UUID(user_id)
        logger.info(f"get_user_bills: user_id is valid UUID: {user_id}")
    except ValueError:
        # If user_id is not a valid UUID format, generate a new one based on the string
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
        logger.info(f"get_user_bills: Generated UUID {user_uuid} from non-UUID user_id: {user_id}")
    
    user_id_str = str(user_uuid)
    logger.info(f"get_user_bills: Using user_id_str: {user_id_str} for lookup")
    
    try:
        # Try to use the database first with robust session management
        try:
            repo = BillAnalysisRepository(session)
            logger.info(f"get_user_bills: Querying database for bills with user_id: {user_uuid}")
            result = await repo.get_user_analyses(user_uuid, limit=limit)
            await session.commit()
            logger.info(f"get_user_bills: Found {len(result)} bills in database for user_id: {user_uuid}")
            return result
        except Exception as db_error:
            # Rollback on any database error
            await session.rollback()
            logger.error(f"get_user_bills: Database error: {db_error}")
            raise db_error
    except Exception as e:
        # Fallback to in-memory storage if database fails
        logger.warning(f"Database operation failed, using in-memory fallback: {e}")
        
        # Log in-memory bills state
        logger.info(f"get_user_bills: In-memory bills users: {list(_in_memory_bills.keys())}")
        
        # Return empty list if user has no bills
        if user_id_str not in _in_memory_bills:
            logger.warning(f"get_user_bills: No bills found in in-memory storage for user_id: {user_id_str}")
            return []
        
        # Convert in-memory bills to BillAnalysis objects
        bills = []
        user_bills = _in_memory_bills[user_id_str]
        logger.info(f"get_user_bills: Found {len(user_bills)} bills in in-memory storage for user_id: {user_id_str}")
        logger.info(f"get_user_bills: Bill IDs: {list(user_bills.keys())}")
        
        for bill_id, bill_data in list(_in_memory_bills[user_id_str].items())[:limit]:
            # Ensure all required attributes exist
            required_attrs = {
                'id': bill_id,
                'filename': bill_data.get('filename', 'unknown.pdf'),
                'created_at': bill_data.get('created_at', ''),
                'status': bill_data.get('status', 'completed'),
                'analysis_type': bill_data.get('analysis_type', 'medical'),
                'total_amount': bill_data.get('total_amount', 0),
                'suspected_overcharges': bill_data.get('suspected_overcharges', 0),
                'confidence_level': bill_data.get('confidence_level', 0)
            }
            
            # Create a mock BillAnalysis object with all required attributes
            mock_bill = type('BillAnalysis', (), {**bill_data, **required_attrs})
            bills.append(mock_bill)
        
        return bills

async def get_bill_by_id(session: AsyncSession, doc_id: str) -> Optional[BillAnalysis]:
    # Handle non-UUID format strings by generating a UUID if needed
    try:
        id_uuid = uuid.UUID(doc_id)
        logger.info(f"get_bill_by_id: doc_id is valid UUID: {doc_id}")
    except ValueError:
        # If doc_id is not a valid UUID format, generate a new one based on the string
        id_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        logger.info(f"get_bill_by_id: Generated UUID {id_uuid} from non-UUID doc_id: {doc_id}")
    
    id_str = str(id_uuid)
    logger.info(f"get_bill_by_id: Using id_str: {id_str} for lookup")
    
    try:
        # Try to use the database first with robust session management
        try:
            repo = BillAnalysisRepository(session)
            logger.info(f"get_bill_by_id: Querying database for bill with id: {id_uuid}")
            result = await repo.get_analysis_by_id(id_uuid)
            await session.commit()
            if result:
                logger.info(f"get_bill_by_id: Found bill in database with id: {id_uuid}")
            else:
                logger.warning(f"get_bill_by_id: No bill found in database with id: {id_uuid}")
            return result
        except Exception as db_error:
            # Rollback on any database error
            await session.rollback()
            logger.error(f"get_bill_by_id: Database error: {db_error}")
            raise db_error
    except Exception as e:
        # Fallback to in-memory storage if database fails
        logger.warning(f"Database operation failed, using in-memory fallback: {e}")
        
        # Log in-memory bills state
        logger.info(f"get_bill_by_id: In-memory bills users: {list(_in_memory_bills.keys())}")
        for user_id, bills in _in_memory_bills.items():
            logger.info(f"get_bill_by_id: User {user_id} has {len(bills)} bills with IDs: {list(bills.keys())}")
        
        # Search for the bill in all users' bills
        for user_id, user_bills in _in_memory_bills.items():
            if id_str in user_bills:
                bill_data = user_bills[id_str]
                # Debug log the retrieved bill data
                logger.info(f"get_bill_by_id: Found bill in in-memory storage for doc_id={id_str} under user_id={user_id}")
                logger.info(f"get_bill_by_id: bill_data keys: {list(bill_data.keys())}")
                
                if 'raw_analysis' in bill_data:
                    raw_analysis = bill_data['raw_analysis']
                    logger.debug(f"raw_analysis type: {type(raw_analysis)}")
                    if isinstance(raw_analysis, dict):
                        logger.debug(f"raw_analysis keys: {list(raw_analysis.keys())}")
                        
                        # Check for line items in various locations
                        if 'final_result' in raw_analysis:
                            logger.debug(f"final_result keys: {list(raw_analysis['final_result'].keys())}")
                            
                        if 'results' in raw_analysis:
                            results = raw_analysis['results']
                            if isinstance(results, dict):
                                logger.debug(f"results keys: {list(results.keys())}")
                                if 'debug_line_items' in results:
                                    logger.debug(f"Found {len(results['debug_line_items'])} debug_line_items")
                
                if 'structured_results' in bill_data:
                    structured_results = bill_data['structured_results']
                    logger.debug(f"structured_results type: {type(structured_results)}")
                    if isinstance(structured_results, dict):
                        logger.debug(f"structured_results keys: {list(structured_results.keys())}")
                        if 'line_items' in structured_results:
                            logger.debug(f"Found {len(structured_results['line_items'])} line_items in structured_results")
                
                # Ensure all required attributes exist
                required_attrs = {
                    'id': id_str,
                    'filename': bill_data.get('filename', 'unknown.pdf'),
                    'created_at': bill_data.get('created_at', ''),
                    'status': bill_data.get('status', 'completed'),
                    'analysis_type': bill_data.get('analysis_type', 'medical'),
                    'total_amount': bill_data.get('total_amount', 0),
                    'suspected_overcharges': bill_data.get('suspected_overcharges', 0),
                    'confidence_level': bill_data.get('confidence_level', 0)
                }
                
                # Create a mock BillAnalysis object with all required attributes
                return type('BillAnalysis', (), {**bill_data, **required_attrs})
        
        # Bill not found
        return None
