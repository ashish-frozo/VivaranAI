#!/usr/bin/env python3
"""
Database Management Script for VivaranAI

This script provides commands for database migrations, backups, and maintenance.
Usage:
    python scripts/manage_db.py migrate    # Run pending migrations
    python scripts/manage_db.py rollback  # Rollback last migration
    python scripts/manage_db.py reset     # Reset database (DANGER!)
    python scripts/manage_db.py backup    # Create database backup
    python scripts/manage_db.py restore   # Restore from backup
    python scripts/manage_db.py status    # Show migration status
"""

import os
import sys
import subprocess
import argparse
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.models import db_manager, create_tables, drop_tables
from config.env_config import config, check_required_config
import structlog

logger = structlog.get_logger(__name__)

class DatabaseManager:
    """Database management operations"""
    
    def __init__(self):
        self.project_root = project_root
        self.alembic_cfg = self.project_root / "alembic.ini"
        
    def run_alembic_command(self, command: str, *args) -> int:
        """Run alembic command with proper configuration"""
        cmd = ["alembic", "-c", str(self.alembic_cfg), command] + list(args)
        
        # Set environment variables
        env = os.environ.copy()
        if hasattr(config, 'database_url'):
            env["DATABASE_URL"] = config.database_url
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, env=env, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.error(f"Alembic command failed: {e}")
            return e.returncode
    
    def migrate(self) -> int:
        """Run pending migrations"""
        logger.info("🔄 Running database migrations...")
        
        # First check if database is accessible
        if not self.check_database_connection():
            return 1
        
        # Run migrations
        result = self.run_alembic_command("upgrade", "head")
        
        if result == 0:
            logger.info("✅ Database migrations completed successfully")
        else:
            logger.error("❌ Database migrations failed")
        
        return result
    
    def rollback(self, steps: int = 1) -> int:
        """Rollback migrations"""
        logger.warning(f"🔄 Rolling back {steps} migration(s)...")
        
        if steps == 1:
            target = "-1"
        else:
            target = f"-{steps}"
        
        result = self.run_alembic_command("downgrade", target)
        
        if result == 0:
            logger.info(f"✅ Rolled back {steps} migration(s)")
        else:
            logger.error("❌ Rollback failed")
        
        return result
    
    def create_migration(self, message: str) -> int:
        """Create a new migration"""
        logger.info(f"📝 Creating new migration: {message}")
        
        result = self.run_alembic_command("revision", "--autogenerate", "-m", message)
        
        if result == 0:
            logger.info("✅ Migration created successfully")
        else:
            logger.error("❌ Migration creation failed")
        
        return result
    
    def show_status(self) -> int:
        """Show current migration status"""
        logger.info("📊 Checking migration status...")
        
        # Show current revision
        self.run_alembic_command("current")
        
        # Show migration history
        logger.info("\n📋 Migration history:")
        return self.run_alembic_command("history", "--verbose")
    
    def reset_database(self, confirm: bool = False) -> int:
        """Reset database - DANGEROUS OPERATION"""
        if not confirm:
            logger.warning("⚠️  This will DELETE ALL DATA in the database!")
            response = input("Type 'RESET' to confirm: ")
            if response != "RESET":
                logger.info("❌ Database reset cancelled")
                return 1
        
        logger.warning("🔥 Resetting database...")
        
        try:
            # Drop all tables
            asyncio.run(drop_tables())
            logger.info("✅ Dropped all tables")
            
            # Recreate tables
            asyncio.run(create_tables())
            logger.info("✅ Recreated all tables")
            
            # Stamp the database with the latest migration
            result = self.run_alembic_command("stamp", "head")
            if result == 0:
                logger.info("✅ Database reset completed")
            else:
                logger.error("❌ Failed to stamp database")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Database reset failed: {e}")
            return 1
    
    def backup_database(self, backup_path: str = None) -> int:
        """Create database backup"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/vivaranai_backup_{timestamp}.sql"
        
        # Create backups directory
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        logger.info(f"💾 Creating database backup: {backup_path}")
        
        # Get database URL
        db_url = getattr(config, 'database_url', 'postgresql://localhost/vivaranai')
        
        # Extract connection parameters
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        
        cmd = [
            "pg_dump",
            f"--host={parsed.hostname or 'localhost'}",
            f"--port={parsed.port or 5432}",
            f"--username={parsed.username or 'postgres'}",
            f"--dbname={parsed.path[1:] if parsed.path else 'vivaranai'}",
            "--verbose",
            "--clean",
            "--no-owner",
            "--no-privileges",
            f"--file={backup_path}"
        ]
        
        env = os.environ.copy()
        if parsed.password:
            env["PGPASSWORD"] = parsed.password
        
        try:
            result = subprocess.run(cmd, env=env, check=True)
            logger.info(f"✅ Database backup created: {backup_path}")
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database backup failed: {e}")
            return 1
        except FileNotFoundError:
            logger.error("❌ pg_dump not found. Please install PostgreSQL client tools.")
            return 1
    
    def restore_database(self, backup_path: str) -> int:
        """Restore database from backup"""
        if not os.path.exists(backup_path):
            logger.error(f"❌ Backup file not found: {backup_path}")
            return 1
        
        logger.warning(f"🔄 Restoring database from: {backup_path}")
        logger.warning("⚠️  This will overwrite existing data!")
        
        response = input("Type 'RESTORE' to confirm: ")
        if response != "RESTORE":
            logger.info("❌ Database restore cancelled")
            return 1
        
        # Get database URL
        db_url = getattr(config, 'database_url', 'postgresql://localhost/vivaranai')
        
        # Extract connection parameters
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        
        cmd = [
            "psql",
            f"--host={parsed.hostname or 'localhost'}",
            f"--port={parsed.port or 5432}",
            f"--username={parsed.username or 'postgres'}",
            f"--dbname={parsed.path[1:] if parsed.path else 'vivaranai'}",
            f"--file={backup_path}"
        ]
        
        env = os.environ.copy()
        if parsed.password:
            env["PGPASSWORD"] = parsed.password
        
        try:
            result = subprocess.run(cmd, env=env, check=True)
            logger.info("✅ Database restored successfully")
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database restore failed: {e}")
            return 1
        except FileNotFoundError:
            logger.error("❌ psql not found. Please install PostgreSQL client tools.")
            return 1
    
    def check_database_connection(self) -> bool:
        """Check if database is accessible"""
        try:
            logger.info("🔍 Checking database connection...")
            result = asyncio.run(db_manager.health_check())
            if result:
                logger.info("✅ Database connection successful")
            else:
                logger.error("❌ Database connection failed")
            return result
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            return False
    
    def init_database(self) -> int:
        """Initialize database with initial migration"""
        logger.info("🚀 Initializing database...")
        
        # Check if migrations directory exists
        migrations_dir = self.project_root / "database" / "migrations" / "versions"
        if not migrations_dir.exists():
            logger.error("❌ Migrations directory not found. Run 'alembic init' first.")
            return 1
        
        # Check if any migrations exist
        migration_files = list(migrations_dir.glob("*.py"))
        if not migration_files:
            logger.info("📝 Creating initial migration...")
            result = self.create_migration("Initial database schema")
            if result != 0:
                return result
        
        # Run migrations
        return self.migrate()

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="VivaranAI Database Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Migrate command
    subparsers.add_parser("migrate", help="Run pending migrations")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--steps", type=int, default=1, help="Number of migrations to rollback")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration message")
    
    # Status command
    subparsers.add_parser("status", help="Show migration status")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database (DANGER!)")
    reset_parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument("--path", help="Backup file path")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_path", help="Path to backup file")
    
    # Init command
    subparsers.add_parser("init", help="Initialize database")
    
    # Check connection command
    subparsers.add_parser("check", help="Check database connection")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check configuration
    if not check_required_config():
        logger.error("❌ Required configuration missing. Please check your .env file.")
        return 1
    
    db_mgr = DatabaseManager()
    
    try:
        if args.command == "migrate":
            return db_mgr.migrate()
        elif args.command == "rollback":
            return db_mgr.rollback(args.steps)
        elif args.command == "create":
            return db_mgr.create_migration(args.message)
        elif args.command == "status":
            return db_mgr.show_status()
        elif args.command == "reset":
            return db_mgr.reset_database(args.confirm)
        elif args.command == "backup":
            return db_mgr.backup_database(args.path)
        elif args.command == "restore":
            return db_mgr.restore_database(args.backup_path)
        elif args.command == "init":
            return db_mgr.init_database()
        elif args.command == "check":
            return 0 if db_mgr.check_database_connection() else 1
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 