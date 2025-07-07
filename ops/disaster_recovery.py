#!/usr/bin/env python3
"""
Disaster Recovery and Backup Management for VivaranAI Production

This module provides comprehensive backup strategies, disaster recovery procedures,
and business continuity planning for production systems.
"""

import asyncio
import boto3
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

logger = structlog.get_logger(__name__)


class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class BackupJob:
    """Represents a backup job"""
    id: str
    name: str
    backup_type: BackupType
    source_path: str
    destination_path: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    size_bytes: int = 0
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'backup_type': self.backup_type.value,
            'source_path': self.source_path,
            'destination_path': self.destination_path,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum,
            'metadata': self.metadata,
            'error_message': self.error_message
        }


class DatabaseBackupManager:
    """Database backup and recovery manager"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.backup_history: List[BackupJob] = []
        
    def create_database_backup(self, backup_name: str = None) -> BackupJob:
        """Create database backup"""
        if not backup_name:
            backup_name = f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_job = BackupJob(
            id=f"db_{int(time.time())}",
            name=backup_name,
            backup_type=BackupType.FULL,
            source_path="database",
            destination_path=f"backups/{backup_name}.sql.gz"
        )
        
        logger.info(f"🗃️  Starting database backup: {backup_name}")
        
        try:
            backup_job.status = BackupStatus.IN_PROGRESS
            backup_job.started_at = datetime.now()
            
            # Create backup directory
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Build pg_dump command
            dump_file = backup_dir / f"{backup_name}.sql"
            cmd = self._build_pg_dump_command(str(dump_file))
            
            # Execute backup
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Compress backup
                compressed_file = self._compress_backup(dump_file)
                backup_job.destination_path = str(compressed_file)
                
                # Calculate checksum and size
                backup_job.checksum = self._calculate_checksum(compressed_file)
                backup_job.size_bytes = compressed_file.stat().st_size
                
                # Clean up uncompressed file
                dump_file.unlink(missing_ok=True)
                
                backup_job.status = BackupStatus.COMPLETED
                backup_job.completed_at = datetime.now()
                
                logger.info(f"✅ Database backup completed: {backup_name}",
                          size_mb=backup_job.size_bytes / (1024 * 1024),
                          checksum=backup_job.checksum[:8])
                
            else:
                backup_job.status = BackupStatus.FAILED
                backup_job.error_message = result.stderr
                logger.error(f"❌ Database backup failed: {result.stderr}")
                
        except Exception as e:
            backup_job.status = BackupStatus.FAILED
            backup_job.error_message = str(e)
            logger.error(f"❌ Database backup failed: {e}")
        
        self.backup_history.append(backup_job)
        return backup_job
    
    def restore_database_backup(self, backup_job: BackupJob) -> bool:
        """Restore database from backup"""
        logger.warning(f"🔄 Starting database restore from: {backup_job.name}")
        
        try:
            backup_path = Path(backup_job.destination_path)
            
            if not backup_path.exists():
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False
            
            # Verify checksum
            if backup_job.checksum and self._calculate_checksum(backup_path) != backup_job.checksum:
                logger.error("❌ Backup file checksum mismatch!")
                return False
            
            # Decompress backup
            sql_file = self._decompress_backup(backup_path)
            
            # Build psql command
            cmd = self._build_psql_command(str(sql_file))
            
            # Execute restore
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up decompressed file
            sql_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Database restored successfully from: {backup_job.name}")
                return True
            else:
                logger.error(f"❌ Database restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database restore failed: {e}")
            return False
    
    def _build_pg_dump_command(self, output_file: str) -> List[str]:
        """Build pg_dump command"""
        cmd = [
            "pg_dump",
            f"--host={self.db_config.get('host', 'localhost')}",
            f"--port={self.db_config.get('port', 5432)}",
            f"--username={self.db_config.get('username', 'postgres')}",
            f"--dbname={self.db_config.get('database', 'vivaranai')}",
            "--verbose",
            "--clean",
            "--no-owner",
            "--no-privileges",
            f"--file={output_file}"
        ]
        return cmd
    
    def _build_psql_command(self, input_file: str) -> List[str]:
        """Build psql command for restore"""
        cmd = [
            "psql",
            f"--host={self.db_config.get('host', 'localhost')}",
            f"--port={self.db_config.get('port', 5432)}",
            f"--username={self.db_config.get('username', 'postgres')}",
            f"--dbname={self.db_config.get('database', 'vivaranai')}",
            f"--file={input_file}"
        ]
        return cmd
    
    def _compress_backup(self, file_path: Path) -> Path:
        """Compress backup file"""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return compressed_path
    
    def _decompress_backup(self, compressed_path: Path) -> Path:
        """Decompress backup file"""
        sql_path = compressed_path.with_suffix('')
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(sql_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return sql_path
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()


class FileSystemBackupManager:
    """File system backup and recovery manager"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_history: List[BackupJob] = []
        
        # Define what to backup
        self.backup_includes = [
            "medbillguardagent/",
            "agents/",
            "config/",
            "data/",
            "security/",
            "deployment/",
            "scripts/",
            ".env",
            "requirements.txt",
            "pyproject.toml",
            "README.md"
        ]
        
        self.backup_excludes = [
            "__pycache__/",
            "*.pyc",
            ".git/",
            ".venv/",
            "venv/",
            "node_modules/",
            "backups/",
            "*.log",
            "htmlcov/",
            ".pytest_cache/"
        ]
    
    def create_filesystem_backup(self, backup_name: str = None, 
                                backup_type: BackupType = BackupType.FULL) -> BackupJob:
        """Create filesystem backup"""
        if not backup_name:
            backup_name = f"fs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_job = BackupJob(
            id=f"fs_{int(time.time())}",
            name=backup_name,
            backup_type=backup_type,
            source_path=str(self.project_root),
            destination_path=f"backups/{backup_name}.tar.gz"
        )
        
        logger.info(f"📁 Starting filesystem backup: {backup_name}")
        
        try:
            backup_job.status = BackupStatus.IN_PROGRESS
            backup_job.started_at = datetime.now()
            
            # Create backup directory
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Create tar archive
            backup_path = backup_dir / f"{backup_name}.tar.gz"
            
            with tarfile.open(backup_path, "w:gz") as tar:
                for include_path in self.backup_includes:
                    full_path = self.project_root / include_path
                    
                    if full_path.exists():
                        # Add to archive with filtering
                        tar.add(full_path, arcname=include_path, 
                               filter=self._tar_filter)
                        logger.debug(f"Added to backup: {include_path}")
            
            # Calculate checksum and size
            backup_job.checksum = self._calculate_checksum(backup_path)
            backup_job.size_bytes = backup_path.stat().st_size
            backup_job.destination_path = str(backup_path)
            
            backup_job.status = BackupStatus.COMPLETED
            backup_job.completed_at = datetime.now()
            
            logger.info(f"✅ Filesystem backup completed: {backup_name}",
                      size_mb=backup_job.size_bytes / (1024 * 1024),
                      checksum=backup_job.checksum[:8])
            
        except Exception as e:
            backup_job.status = BackupStatus.FAILED
            backup_job.error_message = str(e)
            logger.error(f"❌ Filesystem backup failed: {e}")
        
        self.backup_history.append(backup_job)
        return backup_job
    
    def restore_filesystem_backup(self, backup_job: BackupJob, 
                                 restore_path: Path = None) -> bool:
        """Restore filesystem from backup"""
        if not restore_path:
            restore_path = self.project_root
        
        logger.warning(f"🔄 Starting filesystem restore from: {backup_job.name}")
        
        try:
            backup_path = Path(backup_job.destination_path)
            
            if not backup_path.exists():
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False
            
            # Verify checksum
            if backup_job.checksum and self._calculate_checksum(backup_path) != backup_job.checksum:
                logger.error("❌ Backup file checksum mismatch!")
                return False
            
            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(path=restore_path)
            
            logger.info(f"✅ Filesystem restored successfully from: {backup_job.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Filesystem restore failed: {e}")
            return False
    
    def _tar_filter(self, tarinfo):
        """Filter function for tar archive"""
        # Skip excluded patterns
        for exclude_pattern in self.backup_excludes:
            if exclude_pattern.endswith('/'):
                if exclude_pattern[:-1] in tarinfo.name:
                    return None
            elif exclude_pattern.startswith('*'):
                if tarinfo.name.endswith(exclude_pattern[1:]):
                    return None
            elif exclude_pattern in tarinfo.name:
                return None
        
        return tarinfo
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()


class CloudBackupManager:
    """Cloud backup manager for AWS S3, Google Cloud, etc."""
    
    def __init__(self, cloud_config: Dict[str, Any]):
        self.cloud_config = cloud_config
        self.provider = cloud_config.get('provider', 'aws')
        
        if self.provider == 'aws':
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=cloud_config.get('access_key'),
                aws_secret_access_key=cloud_config.get('secret_key'),
                region_name=cloud_config.get('region', 'us-east-1')
            )
            self.bucket_name = cloud_config.get('bucket_name')
    
    def upload_backup_to_cloud(self, backup_job: BackupJob) -> bool:
        """Upload backup to cloud storage"""
        logger.info(f"☁️  Uploading backup to cloud: {backup_job.name}")
        
        try:
            backup_path = Path(backup_job.destination_path)
            
            if not backup_path.exists():
                logger.error(f"❌ Backup file not found: {backup_path}")
                return False
            
            if self.provider == 'aws':
                return self._upload_to_s3(backup_path, backup_job)
            else:
                logger.error(f"❌ Unsupported cloud provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cloud upload failed: {e}")
            return False
    
    def download_backup_from_cloud(self, backup_job: BackupJob) -> bool:
        """Download backup from cloud storage"""
        logger.info(f"☁️  Downloading backup from cloud: {backup_job.name}")
        
        try:
            if self.provider == 'aws':
                return self._download_from_s3(backup_job)
            else:
                logger.error(f"❌ Unsupported cloud provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cloud download failed: {e}")
            return False
    
    def _upload_to_s3(self, backup_path: Path, backup_job: BackupJob) -> bool:
        """Upload backup to AWS S3"""
        try:
            object_key = f"vivaranai-backups/{backup_job.name}"
            
            # Upload with metadata
            extra_args = {
                'Metadata': {
                    'backup-id': backup_job.id,
                    'backup-type': backup_job.backup_type.value,
                    'checksum': backup_job.checksum or '',
                    'created-at': backup_job.created_at.isoformat()
                }
            }
            
            self.s3_client.upload_file(
                str(backup_path),
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )
            
            # Update backup job with cloud location
            backup_job.metadata['cloud_location'] = f"s3://{self.bucket_name}/{object_key}"
            
            logger.info(f"✅ Backup uploaded to S3: s3://{self.bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return False
    
    def _download_from_s3(self, backup_job: BackupJob) -> bool:
        """Download backup from AWS S3"""
        try:
            cloud_location = backup_job.metadata.get('cloud_location')
            
            if not cloud_location or not cloud_location.startswith('s3://'):
                logger.error("❌ Invalid S3 location in backup metadata")
                return False
            
            # Parse S3 location
            s3_path = cloud_location.replace('s3://', '')
            bucket, key = s3_path.split('/', 1)
            
            # Download file
            local_path = Path(backup_job.destination_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(bucket, key, str(local_path))
            
            logger.info(f"✅ Backup downloaded from S3: {cloud_location}")
            return True
            
        except Exception as e:
            logger.error(f"❌ S3 download failed: {e}")
            return False


class DisasterRecoveryManager:
    """Main disaster recovery orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = Path(config.get('project_root', '.'))
        
        # Initialize backup managers
        self.db_backup_manager = DatabaseBackupManager(config.get('database', {}))
        self.fs_backup_manager = FileSystemBackupManager(self.project_root)
        
        # Initialize cloud backup if configured
        self.cloud_backup_manager = None
        if config.get('cloud'):
            self.cloud_backup_manager = CloudBackupManager(config['cloud'])
        
        # Backup schedule configuration
        self.backup_schedule = config.get('backup_schedule', {
            'full_backup_interval_hours': 24,
            'incremental_backup_interval_hours': 6,
            'max_backup_retention_days': 30
        })
        
        # Recovery procedures
        self.recovery_procedures = {
            'database_corruption': self._recover_from_database_corruption,
            'filesystem_corruption': self._recover_from_filesystem_corruption,
            'total_system_failure': self._recover_from_total_system_failure,
            'security_breach': self._recover_from_security_breach
        }
    
    def create_full_backup(self, upload_to_cloud: bool = True) -> Dict[str, BackupJob]:
        """Create complete system backup"""
        logger.info("🔄 Starting full system backup...")
        
        backup_jobs = {}
        
        # Database backup
        logger.info("📊 Creating database backup...")
        db_backup = self.db_backup_manager.create_database_backup()
        backup_jobs['database'] = db_backup
        
        # Filesystem backup
        logger.info("📁 Creating filesystem backup...")
        fs_backup = self.fs_backup_manager.create_filesystem_backup()
        backup_jobs['filesystem'] = fs_backup
        
        # Upload to cloud if configured
        if upload_to_cloud and self.cloud_backup_manager:
            logger.info("☁️  Uploading backups to cloud...")
            
            for backup_type, backup_job in backup_jobs.items():
                if backup_job.status == BackupStatus.COMPLETED:
                    success = self.cloud_backup_manager.upload_backup_to_cloud(backup_job)
                    if success:
                        backup_job.metadata['uploaded_to_cloud'] = True
        
        # Save backup metadata
        self._save_backup_metadata(backup_jobs)
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        logger.info("✅ Full system backup completed",
                   db_status=db_backup.status.value,
                   fs_status=fs_backup.status.value)
        
        return backup_jobs
    
    def execute_recovery_procedure(self, incident_type: str, 
                                  backup_timestamp: datetime = None) -> bool:
        """Execute disaster recovery procedure"""
        logger.warning(f"🚨 Executing disaster recovery for: {incident_type}")
        
        if incident_type not in self.recovery_procedures:
            logger.error(f"❌ Unknown incident type: {incident_type}")
            return False
        
        try:
            # Execute recovery procedure
            recovery_func = self.recovery_procedures[incident_type]
            success = recovery_func(backup_timestamp)
            
            if success:
                logger.info(f"✅ Disaster recovery completed for: {incident_type}")
            else:
                logger.error(f"❌ Disaster recovery failed for: {incident_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Disaster recovery failed: {e}")
            return False
    
    def _recover_from_database_corruption(self, backup_timestamp: datetime = None) -> bool:
        """Recover from database corruption"""
        logger.warning("🔄 Recovering from database corruption...")
        
        # Find latest database backup
        backup_job = self._find_latest_backup('database', backup_timestamp)
        
        if not backup_job:
            logger.error("❌ No suitable database backup found")
            return False
        
        # Download from cloud if necessary
        if backup_job.metadata.get('uploaded_to_cloud') and self.cloud_backup_manager:
            if not Path(backup_job.destination_path).exists():
                self.cloud_backup_manager.download_backup_from_cloud(backup_job)
        
        # Restore database
        return self.db_backup_manager.restore_database_backup(backup_job)
    
    def _recover_from_filesystem_corruption(self, backup_timestamp: datetime = None) -> bool:
        """Recover from filesystem corruption"""
        logger.warning("🔄 Recovering from filesystem corruption...")
        
        # Find latest filesystem backup
        backup_job = self._find_latest_backup('filesystem', backup_timestamp)
        
        if not backup_job:
            logger.error("❌ No suitable filesystem backup found")
            return False
        
        # Download from cloud if necessary
        if backup_job.metadata.get('uploaded_to_cloud') and self.cloud_backup_manager:
            if not Path(backup_job.destination_path).exists():
                self.cloud_backup_manager.download_backup_from_cloud(backup_job)
        
        # Create backup of current state before restore
        current_backup = self.fs_backup_manager.create_filesystem_backup(
            f"pre_restore_backup_{int(time.time())}"
        )
        
        # Restore filesystem
        return self.fs_backup_manager.restore_filesystem_backup(backup_job)
    
    def _recover_from_total_system_failure(self, backup_timestamp: datetime = None) -> bool:
        """Recover from total system failure"""
        logger.warning("🔄 Recovering from total system failure...")
        
        # Recover both database and filesystem
        db_success = self._recover_from_database_corruption(backup_timestamp)
        fs_success = self._recover_from_filesystem_corruption(backup_timestamp)
        
        return db_success and fs_success
    
    def _recover_from_security_breach(self, backup_timestamp: datetime = None) -> bool:
        """Recover from security breach"""
        logger.warning("🔄 Recovering from security breach...")
        
        # This is more complex - might need to restore to a known good state
        # and then apply security patches
        
        # Find backup from before the breach
        if not backup_timestamp:
            # Use backup from 24 hours ago as a safe point
            backup_timestamp = datetime.now() - timedelta(hours=24)
        
        # Full system restore
        return self._recover_from_total_system_failure(backup_timestamp)
    
    def _find_latest_backup(self, backup_type: str, 
                           before_timestamp: datetime = None) -> Optional[BackupJob]:
        """Find latest backup of given type"""
        if backup_type == 'database':
            backups = self.db_backup_manager.backup_history
        elif backup_type == 'filesystem':
            backups = self.fs_backup_manager.backup_history
        else:
            return None
        
        # Filter by timestamp if provided
        if before_timestamp:
            backups = [b for b in backups if b.completed_at and b.completed_at <= before_timestamp]
        
        # Filter completed backups only
        completed_backups = [b for b in backups if b.status == BackupStatus.COMPLETED]
        
        if not completed_backups:
            return None
        
        # Return latest backup
        return max(completed_backups, key=lambda b: b.completed_at)
    
    def _save_backup_metadata(self, backup_jobs: Dict[str, BackupJob]):
        """Save backup metadata to file"""
        metadata_file = self.project_root / "backups" / "backup_metadata.json"
        metadata_file.parent.mkdir(exist_ok=True)
        
        # Load existing metadata
        existing_metadata = []
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception:
                existing_metadata = []
        
        # Add new backup jobs
        for backup_job in backup_jobs.values():
            existing_metadata.append(backup_job.to_dict())
        
        # Sort by timestamp
        existing_metadata.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        retention_days = self.backup_schedule['max_backup_retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        backup_dir = Path("backups")
        if not backup_dir.exists():
            return
        
        # Clean up local backup files
        for backup_file in backup_dir.glob("*.tar.gz"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    backup_file.unlink()
                    logger.info(f"🗑️  Cleaned up old backup: {backup_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not delete old backup {backup_file}: {e}")
        
        for backup_file in backup_dir.glob("*.sql.gz"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    backup_file.unlink()
                    logger.info(f"🗑️  Cleaned up old backup: {backup_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not delete old backup {backup_file}: {e}")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status"""
        latest_db_backup = self._find_latest_backup('database')
        latest_fs_backup = self._find_latest_backup('filesystem')
        
        return {
            'last_full_backup': {
                'database': latest_db_backup.to_dict() if latest_db_backup else None,
                'filesystem': latest_fs_backup.to_dict() if latest_fs_backup else None
            },
            'backup_counts': {
                'database': len(self.db_backup_manager.backup_history),
                'filesystem': len(self.fs_backup_manager.backup_history)
            },
            'cloud_backup_enabled': self.cloud_backup_manager is not None,
            'backup_schedule': self.backup_schedule,
            'available_recovery_procedures': list(self.recovery_procedures.keys())
        }


def main():
    """CLI interface for disaster recovery"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VivaranAI Disaster Recovery Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--type", choices=["full", "database", "filesystem"], 
                              default="full", help="Backup type")
    backup_parser.add_argument("--no-cloud", action="store_true", help="Skip cloud upload")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    
    # Recovery command
    recovery_parser = subparsers.add_parser("recover", help="Execute disaster recovery")
    recovery_parser.add_argument("incident_type", 
                               choices=["database_corruption", "filesystem_corruption", 
                                       "total_system_failure", "security_breach"],
                               help="Type of incident")
    recovery_parser.add_argument("--backup-timestamp", help="Backup timestamp (ISO format)")
    
    # Status command
    subparsers.add_parser("status", help="Show backup status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Load configuration
    config = {
        'project_root': '.',
        'database': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'username': os.getenv('DB_USER', 'postgres'),
            'database': os.getenv('DB_NAME', 'vivaranai')
        }
    }
    
    # Add cloud config if available
    if os.getenv('AWS_ACCESS_KEY_ID'):
        config['cloud'] = {
            'provider': 'aws',
            'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            'bucket_name': os.getenv('S3_BACKUP_BUCKET')
        }
    
    dr_manager = DisasterRecoveryManager(config)
    
    try:
        if args.command == "backup":
            if args.type == "full":
                jobs = dr_manager.create_full_backup(not args.no_cloud)
                print(f"✅ Full backup completed: {len(jobs)} jobs")
            elif args.type == "database":
                job = dr_manager.db_backup_manager.create_database_backup()
                print(f"✅ Database backup: {job.status.value}")
            elif args.type == "filesystem":
                job = dr_manager.fs_backup_manager.create_filesystem_backup()
                print(f"✅ Filesystem backup: {job.status.value}")
        
        elif args.command == "status":
            status = dr_manager.get_backup_status()
            print(json.dumps(status, indent=2))
        
        elif args.command == "recover":
            backup_timestamp = None
            if args.backup_timestamp:
                backup_timestamp = datetime.fromisoformat(args.backup_timestamp)
            
            success = dr_manager.execute_recovery_procedure(args.incident_type, backup_timestamp)
            return 0 if success else 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 