"""
Database utility functions for chat context and bill analysis persistence.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import BillAnalysis, BillAnalysisRepository

# In-memory storage fallback when database is not available
_in_memory_bills = {}
logger = logging.getLogger(__name__)

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
            
            # Create analysis with explicit transaction handling
            result = await repo.create_analysis(analysis_data)
            await session.commit()
            return result
        except Exception as db_error:
            # Rollback on any database error
            await session.rollback()
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
    except ValueError:
        # If user_id is not a valid UUID format, generate a new one based on the string
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
    
    user_id_str = str(user_uuid)
    
    try:
        # Try to use the database first with robust session management
        try:
            repo = BillAnalysisRepository(session)
            result = await repo.get_user_analyses(user_uuid, limit=limit)
            await session.commit()
            return result
        except Exception as db_error:
            # Rollback on any database error
            await session.rollback()
            raise db_error
    except Exception as e:
        # Fallback to in-memory storage if database fails
        logger.warning(f"Database operation failed, using in-memory fallback: {e}")
        
        # Return empty list if user has no bills
        if user_id_str not in _in_memory_bills:
            return []
        
        # Convert in-memory bills to BillAnalysis objects
        bills = []
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
    except ValueError:
        # If doc_id is not a valid UUID format, generate a new one based on the string
        id_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
    
    id_str = str(id_uuid)
    
    try:
        # Try to use the database first with robust session management
        try:
            repo = BillAnalysisRepository(session)
            result = await repo.get_analysis_by_id(id_uuid)
            await session.commit()
            return result
        except Exception as db_error:
            # Rollback on any database error
            await session.rollback()
            raise db_error
    except Exception as e:
        # Fallback to in-memory storage if database fails
        logger.warning(f"Database operation failed, using in-memory fallback: {e}")
        
        # Search for the bill in all users' bills
        for user_id, user_bills in _in_memory_bills.items():
            if id_str in user_bills:
                bill_data = user_bills[id_str]
                # Debug log the retrieved bill data
                logger.debug(f"Retrieved bill data for doc_id={id_str}")
                logger.debug(f"bill_data keys: {list(bill_data.keys())}")
                
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
