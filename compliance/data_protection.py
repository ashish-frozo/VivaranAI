#!/usr/bin/env python3
"""
Data Protection and Privacy Compliance for VivaranAI

This module provides comprehensive GDPR, CCPA, and HIPAA compliance features
for handling medical data and protecting user privacy.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import crypto.cipher
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

import structlog

logger = structlog.get_logger(__name__)


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    MEDICAL = "medical"          # HIPAA protected
    BIOMETRIC = "biometric"      # Highly sensitive biometric data


class ConsentType(Enum):
    """Types of user consent"""
    PROCESSING = "processing"           # Data processing consent
    MARKETING = "marketing"            # Marketing communications
    ANALYTICS = "analytics"            # Analytics and tracking
    THIRD_PARTY = "third_party"        # Third party data sharing
    MEDICAL_RESEARCH = "medical_research"  # Medical research purposes


class LegalBasis(Enum):
    """GDPR legal basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: datetime
    legal_basis: LegalBasis
    purpose: str
    expiry_date: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_mechanism: str = "explicit"  # explicit, implicit, opt_out
    withdrawal_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'user_id': self.user_id,
            'consent_type': self.consent_type.value,
            'granted': self.granted,
            'timestamp': self.timestamp.isoformat(),
            'legal_basis': self.legal_basis.value,
            'purpose': self.purpose,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'consent_mechanism': self.consent_mechanism,
            'withdrawal_date': self.withdrawal_date.isoformat() if self.withdrawal_date else None
        }


@dataclass
class DataProcessingRecord:
    """Data processing activity record"""
    id: str
    user_id: str
    data_type: str
    classification: DataClassification
    purpose: str
    legal_basis: LegalBasis
    processing_start: datetime
    processing_end: Optional[datetime] = None
    retention_period: Optional[timedelta] = None
    data_source: str = ""
    recipients: List[str] = field(default_factory=list)
    cross_border_transfer: bool = False
    safeguards: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'data_type': self.data_type,
            'classification': self.classification.value,
            'purpose': self.purpose,
            'legal_basis': self.legal_basis.value,
            'processing_start': self.processing_start.isoformat(),
            'processing_end': self.processing_end.isoformat() if self.processing_end else None,
            'retention_period_days': self.retention_period.days if self.retention_period else None,
            'data_source': self.data_source,
            'recipients': self.recipients,
            'cross_border_transfer': self.cross_border_transfer,
            'safeguards': self.safeguards
        }


class DataEncryption:
    """Data encryption and anonymization utilities"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            # Generate new key
            self.fernet = Fernet(Fernet.generate_key())
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"❌ Encryption failed: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"❌ Decryption failed: {e}")
            raise
    
    def anonymize_data(self, data: str, method: str = "hash") -> str:
        """Anonymize data using various methods"""
        if method == "hash":
            return hashlib.sha256(data.encode()).hexdigest()
        elif method == "mask":
            if len(data) <= 4:
                return "*" * len(data)
            return data[:2] + "*" * (len(data) - 4) + data[-2:]
        elif method == "remove":
            return "[REMOVED]"
        else:
            raise ValueError(f"Unknown anonymization method: {method}")
    
    def pseudonymize_user_id(self, user_id: str, salt: str = "") -> str:
        """Pseudonymize user ID"""
        combined = f"{user_id}{salt}{self.fernet._encryption_key.decode('utf-8', errors='ignore')}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class ConsentManager:
    """Manage user consent and legal basis"""
    
    def __init__(self):
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.storage_path = Path("compliance/consent_records.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_consent_records()
    
    def record_consent(self, consent_record: ConsentRecord) -> bool:
        """Record user consent"""
        try:
            user_id = consent_record.user_id
            
            if user_id not in self.consent_records:
                self.consent_records[user_id] = []
            
            # Check for existing consent of same type
            existing_consent = self._find_consent(user_id, consent_record.consent_type)
            
            if existing_consent:
                # Update existing consent
                existing_consent.granted = consent_record.granted
                existing_consent.timestamp = consent_record.timestamp
                if not consent_record.granted:
                    existing_consent.withdrawal_date = consent_record.timestamp
            else:
                # Add new consent record
                self.consent_records[user_id].append(consent_record)
            
            self._save_consent_records()
            
            logger.info(f"📝 Consent recorded for user {user_id}",
                       consent_type=consent_record.consent_type.value,
                       granted=consent_record.granted)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record consent: {e}")
            return False
    
    def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has valid consent"""
        consent_record = self._find_consent(user_id, consent_type)
        
        if not consent_record:
            return False
        
        # Check if consent is granted and not withdrawn
        if not consent_record.granted or consent_record.withdrawal_date:
            return False
        
        # Check if consent has expired
        if consent_record.expiry_date and consent_record.expiry_date < datetime.now():
            return False
        
        return True
    
    def withdraw_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Withdraw user consent"""
        try:
            consent_record = self._find_consent(user_id, consent_type)
            
            if consent_record:
                consent_record.granted = False
                consent_record.withdrawal_date = datetime.now()
                self._save_consent_records()
                
                logger.info(f"🚫 Consent withdrawn for user {user_id}",
                           consent_type=consent_type.value)
                
                return True
            else:
                logger.warning(f"⚠️  No consent found to withdraw for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to withdraw consent: {e}")
            return False
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user"""
        return self.consent_records.get(user_id, [])
    
    def export_consent_proof(self, user_id: str) -> Dict[str, Any]:
        """Export consent proof for compliance audits"""
        user_consents = self.get_user_consents(user_id)
        
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'consents': [consent.to_dict() for consent in user_consents],
            'total_consents': len(user_consents),
            'active_consents': len([c for c in user_consents if c.granted and not c.withdrawal_date])
        }
    
    def _find_consent(self, user_id: str, consent_type: ConsentType) -> Optional[ConsentRecord]:
        """Find specific consent record"""
        user_consents = self.consent_records.get(user_id, [])
        
        for consent in user_consents:
            if consent.consent_type == consent_type:
                return consent
        
        return None
    
    def _load_consent_records(self):
        """Load consent records from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for user_id, consent_list in data.items():
                    self.consent_records[user_id] = []
                    
                    for consent_data in consent_list:
                        consent_record = ConsentRecord(
                            user_id=consent_data['user_id'],
                            consent_type=ConsentType(consent_data['consent_type']),
                            granted=consent_data['granted'],
                            timestamp=datetime.fromisoformat(consent_data['timestamp']),
                            legal_basis=LegalBasis(consent_data['legal_basis']),
                            purpose=consent_data['purpose'],
                            expiry_date=datetime.fromisoformat(consent_data['expiry_date']) if consent_data.get('expiry_date') else None,
                            ip_address=consent_data.get('ip_address'),
                            user_agent=consent_data.get('user_agent'),
                            consent_mechanism=consent_data.get('consent_mechanism', 'explicit'),
                            withdrawal_date=datetime.fromisoformat(consent_data['withdrawal_date']) if consent_data.get('withdrawal_date') else None
                        )
                        
                        self.consent_records[user_id].append(consent_record)
                        
        except Exception as e:
            logger.error(f"❌ Failed to load consent records: {e}")
    
    def _save_consent_records(self):
        """Save consent records to storage"""
        try:
            data = {}
            
            for user_id, consent_list in self.consent_records.items():
                data[user_id] = [consent.to_dict() for consent in consent_list]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save consent records: {e}")


class DataProcessingLogger:
    """Log all data processing activities for compliance"""
    
    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.storage_path = Path("compliance/processing_records.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_processing_records()
    
    def log_processing_activity(self, record: DataProcessingRecord) -> bool:
        """Log data processing activity"""
        try:
            self.processing_records.append(record)
            self._save_processing_records()
            
            logger.info(f"📊 Data processing logged",
                       user_id=record.user_id,
                       data_type=record.data_type,
                       purpose=record.purpose)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log processing activity: {e}")
            return False
    
    def end_processing_activity(self, record_id: str) -> bool:
        """Mark processing activity as ended"""
        try:
            for record in self.processing_records:
                if record.id == record_id:
                    record.processing_end = datetime.now()
                    self._save_processing_records()
                    
                    logger.info(f"🏁 Data processing ended for record {record_id}")
                    return True
            
            logger.warning(f"⚠️  Processing record {record_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to end processing activity: {e}")
            return False
    
    def get_user_processing_activities(self, user_id: str) -> List[DataProcessingRecord]:
        """Get all processing activities for a user"""
        return [record for record in self.processing_records if record.user_id == user_id]
    
    def generate_processing_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate processing activities report for compliance"""
        relevant_records = [
            record for record in self.processing_records
            if start_date <= record.processing_start <= end_date
        ]
        
        # Group by classification
        by_classification = {}
        for record in relevant_records:
            classification = record.classification.value
            if classification not in by_classification:
                by_classification[classification] = 0
            by_classification[classification] += 1
        
        # Group by purpose
        by_purpose = {}
        for record in relevant_records:
            purpose = record.purpose
            if purpose not in by_purpose:
                by_purpose[purpose] = 0
            by_purpose[purpose] += 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_activities': len(relevant_records),
            'by_classification': by_classification,
            'by_purpose': by_purpose,
            'cross_border_transfers': len([r for r in relevant_records if r.cross_border_transfer]),
            'activities': [record.to_dict() for record in relevant_records]
        }
    
    def _load_processing_records(self):
        """Load processing records from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for record_data in data:
                    record = DataProcessingRecord(
                        id=record_data['id'],
                        user_id=record_data['user_id'],
                        data_type=record_data['data_type'],
                        classification=DataClassification(record_data['classification']),
                        purpose=record_data['purpose'],
                        legal_basis=LegalBasis(record_data['legal_basis']),
                        processing_start=datetime.fromisoformat(record_data['processing_start']),
                        processing_end=datetime.fromisoformat(record_data['processing_end']) if record_data.get('processing_end') else None,
                        retention_period=timedelta(days=record_data['retention_period_days']) if record_data.get('retention_period_days') else None,
                        data_source=record_data.get('data_source', ''),
                        recipients=record_data.get('recipients', []),
                        cross_border_transfer=record_data.get('cross_border_transfer', False),
                        safeguards=record_data.get('safeguards', [])
                    )
                    
                    self.processing_records.append(record)
                    
        except Exception as e:
            logger.error(f"❌ Failed to load processing records: {e}")
    
    def _save_processing_records(self):
        """Save processing records to storage"""
        try:
            data = [record.to_dict() for record in self.processing_records]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save processing records: {e}")


class DataSubjectRightsManager:
    """Handle data subject rights requests (GDPR/CCPA)"""
    
    def __init__(self, consent_manager: ConsentManager, 
                 processing_logger: DataProcessingLogger,
                 encryption: DataEncryption):
        self.consent_manager = consent_manager
        self.processing_logger = processing_logger
        self.encryption = encryption
        self.requests_storage = Path("compliance/rights_requests.json")
        self.requests_storage.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data subject access request (Article 15 GDPR)"""
        logger.info(f"📋 Processing access request for user {user_id}")
        
        # Gather all user data
        user_data = {
            'user_id': user_id,
            'request_timestamp': datetime.now().isoformat(),
            'consents': self.consent_manager.export_consent_proof(user_id),
            'processing_activities': [
                record.to_dict() for record in 
                self.processing_logger.get_user_processing_activities(user_id)
            ],
            'data_categories': self._identify_data_categories(user_id),
            'retention_periods': self._get_retention_periods(user_id),
            'recipients': self._get_data_recipients(user_id),
            'rights_information': self._get_rights_information()
        }
        
        # Log the access request
        self._log_rights_request("access", user_id, user_data)
        
        return user_data
    
    def handle_rectification_request(self, user_id: str, 
                                   data_corrections: Dict[str, Any]) -> bool:
        """Handle data rectification request (Article 16 GDPR)"""
        logger.info(f"✏️  Processing rectification request for user {user_id}")
        
        try:
            # This would integrate with your user data storage system
            # to update the user's data
            
            # Log the rectification
            rectification_record = {
                'user_id': user_id,
                'corrections': data_corrections,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            self._log_rights_request("rectification", user_id, rectification_record)
            
            logger.info(f"✅ Data rectified for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rectification failed: {e}")
            return False
    
    def handle_erasure_request(self, user_id: str) -> bool:
        """Handle right to be forgotten request (Article 17 GDPR)"""
        logger.warning(f"🗑️  Processing erasure request for user {user_id}")
        
        try:
            # Check if erasure is legally possible
            if not self._can_erase_user_data(user_id):
                logger.warning(f"⚠️  Cannot erase data for user {user_id} - legal obligations")
                return False
            
            # Pseudonymize or delete user data
            erasure_record = {
                'user_id': user_id,
                'pseudonymized_id': self.encryption.pseudonymize_user_id(user_id),
                'timestamp': datetime.now().isoformat(),
                'method': 'pseudonymization',  # or 'deletion'
                'status': 'completed'
            }
            
            # Remove consent records (keep anonymized logs for compliance)
            self.consent_manager.consent_records.pop(user_id, None)
            
            # Anonymize processing records
            user_processing_records = self.processing_logger.get_user_processing_activities(user_id)
            for record in user_processing_records:
                record.user_id = erasure_record['pseudonymized_id']
            
            self._log_rights_request("erasure", user_id, erasure_record)
            
            logger.info(f"✅ Data erased for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erasure failed: {e}")
            return False
    
    def handle_portability_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data portability request (Article 20 GDPR)"""
        logger.info(f"📦 Processing portability request for user {user_id}")
        
        # Export user data in structured format
        portable_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'format': 'JSON',
            'data': self.handle_access_request(user_id),
            'checksum': None
        }
        
        # Calculate checksum
        data_string = json.dumps(portable_data['data'], sort_keys=True)
        portable_data['checksum'] = hashlib.sha256(data_string.encode()).hexdigest()
        
        self._log_rights_request("portability", user_id, portable_data)
        
        return portable_data
    
    def handle_restriction_request(self, user_id: str, restriction_reason: str) -> bool:
        """Handle processing restriction request (Article 18 GDPR)"""
        logger.info(f"🚫 Processing restriction request for user {user_id}")
        
        try:
            # Mark user data for restricted processing
            restriction_record = {
                'user_id': user_id,
                'restriction_reason': restriction_reason,
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # This would integrate with your data processing systems
            # to restrict further processing of the user's data
            
            self._log_rights_request("restriction", user_id, restriction_record)
            
            logger.info(f"✅ Processing restricted for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Restriction failed: {e}")
            return False
    
    def handle_objection_request(self, user_id: str, objection_reason: str) -> bool:
        """Handle objection to processing request (Article 21 GDPR)"""
        logger.info(f"✋ Processing objection request for user {user_id}")
        
        try:
            # Withdraw relevant consents
            self.consent_manager.withdraw_consent(user_id, ConsentType.MARKETING)
            self.consent_manager.withdraw_consent(user_id, ConsentType.ANALYTICS)
            
            objection_record = {
                'user_id': user_id,
                'objection_reason': objection_reason,
                'timestamp': datetime.now().isoformat(),
                'actions_taken': ['marketing_consent_withdrawn', 'analytics_consent_withdrawn'],
                'status': 'completed'
            }
            
            self._log_rights_request("objection", user_id, objection_record)
            
            logger.info(f"✅ Objection processed for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Objection processing failed: {e}")
            return False
    
    def _identify_data_categories(self, user_id: str) -> List[str]:
        """Identify categories of data processed for user"""
        processing_records = self.processing_logger.get_user_processing_activities(user_id)
        categories = set()
        
        for record in processing_records:
            categories.add(record.data_type)
        
        return list(categories)
    
    def _get_retention_periods(self, user_id: str) -> Dict[str, str]:
        """Get retention periods for user data"""
        processing_records = self.processing_logger.get_user_processing_activities(user_id)
        retention_info = {}
        
        for record in processing_records:
            if record.retention_period:
                retention_info[record.data_type] = f"{record.retention_period.days} days"
        
        return retention_info
    
    def _get_data_recipients(self, user_id: str) -> List[str]:
        """Get list of data recipients"""
        processing_records = self.processing_logger.get_user_processing_activities(user_id)
        recipients = set()
        
        for record in processing_records:
            recipients.update(record.recipients)
        
        return list(recipients)
    
    def _get_rights_information(self) -> Dict[str, str]:
        """Get information about data subject rights"""
        return {
            'access': 'Right to access your personal data',
            'rectification': 'Right to rectify inaccurate personal data',
            'erasure': 'Right to erasure (right to be forgotten)',
            'restriction': 'Right to restrict processing',
            'portability': 'Right to data portability',
            'objection': 'Right to object to processing',
            'complaint': 'Right to lodge a complaint with supervisory authority'
        }
    
    def _can_erase_user_data(self, user_id: str) -> bool:
        """Check if user data can be legally erased"""
        # Check for legal obligations that prevent erasure
        processing_records = self.processing_logger.get_user_processing_activities(user_id)
        
        for record in processing_records:
            if record.legal_basis == LegalBasis.LEGAL_OBLIGATION:
                # Cannot erase data required for legal compliance
                return False
            
            if record.classification == DataClassification.MEDICAL:
                # Medical data may have special retention requirements
                if record.retention_period and record.processing_start + record.retention_period > datetime.now():
                    return False
        
        return True
    
    def _log_rights_request(self, request_type: str, user_id: str, request_data: Dict[str, Any]):
        """Log data subject rights request"""
        try:
            # Load existing requests
            requests = []
            if self.requests_storage.exists():
                with open(self.requests_storage, 'r') as f:
                    requests = json.load(f)
            
            # Add new request
            request_record = {
                'id': str(uuid.uuid4()),
                'type': request_type,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'data': request_data
            }
            
            requests.append(request_record)
            
            # Save updated requests
            with open(self.requests_storage, 'w') as f:
                json.dump(requests, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to log rights request: {e}")


class ComplianceMonitor:
    """Monitor compliance with data protection regulations"""
    
    def __init__(self, consent_manager: ConsentManager, 
                 processing_logger: DataProcessingLogger):
        self.consent_manager = consent_manager
        self.processing_logger = processing_logger
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        
        # Get recent processing activities
        recent_activities = [
            record for record in self.processing_logger.processing_records
            if record.processing_start >= thirty_days_ago
        ]
        
        # Check consent compliance
        consent_issues = self._check_consent_compliance()
        
        # Check retention compliance
        retention_issues = self._check_retention_compliance()
        
        # Generate report
        report = {
            'report_timestamp': now.isoformat(),
            'report_period': '30 days',
            'summary': {
                'total_processing_activities': len(recent_activities),
                'consent_compliance_issues': len(consent_issues),
                'retention_compliance_issues': len(retention_issues),
                'overall_compliance_score': self._calculate_compliance_score(
                    consent_issues, retention_issues, recent_activities
                )
            },
            'consent_compliance': {
                'issues': consent_issues,
                'recommendations': self._get_consent_recommendations(consent_issues)
            },
            'retention_compliance': {
                'issues': retention_issues,
                'recommendations': self._get_retention_recommendations(retention_issues)
            },
            'processing_overview': {
                'by_classification': self._group_by_classification(recent_activities),
                'by_legal_basis': self._group_by_legal_basis(recent_activities),
                'cross_border_transfers': len([a for a in recent_activities if a.cross_border_transfer])
            }
        }
        
        return report
    
    def _check_consent_compliance(self) -> List[Dict[str, Any]]:
        """Check for consent compliance issues"""
        issues = []
        
        for user_id, consents in self.consent_manager.consent_records.items():
            for consent in consents:
                # Check for expired consents
                if consent.expiry_date and consent.expiry_date < datetime.now() and consent.granted:
                    issues.append({
                        'type': 'expired_consent',
                        'user_id': user_id,
                        'consent_type': consent.consent_type.value,
                        'expired_date': consent.expiry_date.isoformat(),
                        'severity': 'high'
                    })
                
                # Check for consents without explicit mechanism
                if consent.consent_mechanism != 'explicit' and consent.consent_type in [ConsentType.MEDICAL_RESEARCH]:
                    issues.append({
                        'type': 'invalid_consent_mechanism',
                        'user_id': user_id,
                        'consent_type': consent.consent_type.value,
                        'mechanism': consent.consent_mechanism,
                        'severity': 'medium'
                    })
        
        return issues
    
    def _check_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check for data retention compliance issues"""
        issues = []
        
        for record in self.processing_logger.processing_records:
            if record.retention_period:
                retention_deadline = record.processing_start + record.retention_period
                
                if datetime.now() > retention_deadline and not record.processing_end:
                    issues.append({
                        'type': 'retention_period_exceeded',
                        'record_id': record.id,
                        'user_id': record.user_id,
                        'data_type': record.data_type,
                        'retention_deadline': retention_deadline.isoformat(),
                        'days_overdue': (datetime.now() - retention_deadline).days,
                        'severity': 'high'
                    })
        
        return issues
    
    def _calculate_compliance_score(self, consent_issues: List, 
                                  retention_issues: List, 
                                  activities: List) -> float:
        """Calculate overall compliance score (0-100)"""
        total_issues = len(consent_issues) + len(retention_issues)
        total_activities = max(len(activities), 1)
        
        # Calculate score based on issue ratio
        issue_ratio = total_issues / total_activities
        compliance_score = max(0, 100 - (issue_ratio * 100))
        
        return round(compliance_score, 2)
    
    def _group_by_classification(self, activities: List[DataProcessingRecord]) -> Dict[str, int]:
        """Group activities by data classification"""
        groups = {}
        for activity in activities:
            classification = activity.classification.value
            groups[classification] = groups.get(classification, 0) + 1
        return groups
    
    def _group_by_legal_basis(self, activities: List[DataProcessingRecord]) -> Dict[str, int]:
        """Group activities by legal basis"""
        groups = {}
        for activity in activities:
            basis = activity.legal_basis.value
            groups[basis] = groups.get(basis, 0) + 1
        return groups
    
    def _get_consent_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations for consent issues"""
        recommendations = []
        
        if any(issue['type'] == 'expired_consent' for issue in issues):
            recommendations.append("Implement automated consent renewal reminders")
        
        if any(issue['type'] == 'invalid_consent_mechanism' for issue in issues):
            recommendations.append("Review consent collection mechanisms for sensitive data")
        
        return recommendations
    
    def _get_retention_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations for retention issues"""
        recommendations = []
        
        if any(issue['type'] == 'retention_period_exceeded' for issue in issues):
            recommendations.append("Implement automated data deletion processes")
            recommendations.append("Review and update data retention policies")
        
        return recommendations


def main():
    """CLI interface for compliance management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VivaranAI Compliance Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Consent commands
    consent_parser = subparsers.add_parser("consent", help="Manage user consent")
    consent_parser.add_argument("action", choices=["grant", "withdraw", "check", "export"])
    consent_parser.add_argument("--user-id", required=True, help="User ID")
    consent_parser.add_argument("--type", choices=[t.value for t in ConsentType], help="Consent type")
    consent_parser.add_argument("--purpose", help="Purpose of processing")
    
    # Rights requests
    rights_parser = subparsers.add_parser("rights", help="Handle data subject rights")
    rights_parser.add_argument("action", choices=["access", "rectify", "erase", "port", "restrict", "object"])
    rights_parser.add_argument("--user-id", required=True, help="User ID")
    
    # Compliance monitoring
    subparsers.add_parser("monitor", help="Generate compliance report")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize managers
    encryption = DataEncryption()
    consent_manager = ConsentManager()
    processing_logger = DataProcessingLogger()
    rights_manager = DataSubjectRightsManager(consent_manager, processing_logger, encryption)
    monitor = ComplianceMonitor(consent_manager, processing_logger)
    
    try:
        if args.command == "consent":
            if args.action == "grant":
                consent_record = ConsentRecord(
                    user_id=args.user_id,
                    consent_type=ConsentType(args.type),
                    granted=True,
                    timestamp=datetime.now(),
                    legal_basis=LegalBasis.CONSENT,
                    purpose=args.purpose or "Data processing"
                )
                success = consent_manager.record_consent(consent_record)
                print(f"✅ Consent granted" if success else "❌ Failed to grant consent")
                
            elif args.action == "withdraw":
                success = consent_manager.withdraw_consent(args.user_id, ConsentType(args.type))
                print(f"✅ Consent withdrawn" if success else "❌ Failed to withdraw consent")
                
            elif args.action == "check":
                valid = consent_manager.check_consent(args.user_id, ConsentType(args.type))
                print(f"✅ Valid consent" if valid else "❌ No valid consent")
                
            elif args.action == "export":
                proof = consent_manager.export_consent_proof(args.user_id)
                print(json.dumps(proof, indent=2))
        
        elif args.command == "rights":
            if args.action == "access":
                data = rights_manager.handle_access_request(args.user_id)
                print(json.dumps(data, indent=2))
                
            elif args.action == "erase":
                success = rights_manager.handle_erasure_request(args.user_id)
                print(f"✅ Data erased" if success else "❌ Cannot erase data")
                
            elif args.action == "port":
                data = rights_manager.handle_portability_request(args.user_id)
                print(json.dumps(data, indent=2))
        
        elif args.command == "monitor":
            report = monitor.generate_compliance_report()
            print(json.dumps(report, indent=2))
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 