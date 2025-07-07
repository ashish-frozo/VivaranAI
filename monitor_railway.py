#!/usr/bin/env python3
"""
Railway Deployment Monitor
Automatically monitors Railway logs and detects errors for immediate fixing.
"""

import subprocess
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

class RailwayMonitor:
    """Monitors Railway deployment logs and detects common error patterns."""
    
    def __init__(self):
        self.error_patterns = {
            'module_not_found': r'ModuleNotFoundError: No module named \'(\w+)\'',
            'import_error': r'ImportError: (.+)',
            'attribute_error': r'AttributeError: (.+)',
            'type_error': r'TypeError: (.+)',
            'connection_error': r'ConnectionError: (.+)',
            'redis_error': r'redis\.exceptions\.(.+)',
            'database_error': r'DatabaseError: (.+)',
            'build_failed': r'Build failed',
            'deployment_failed': r'Deploy failed',
            'health_check_failed': r'Healthcheck failed',
            'startup_failed': r'Application startup failed',
        }
        
        self.fixes_applied = set()
        
    def get_railway_logs(self) -> Optional[str]:
        """Get current Railway logs."""
        try:
            result = subprocess.run(
                ['railway', 'logs', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                # Try without --json flag
                result = subprocess.run(
                    ['railway', 'logs'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.stdout if result.returncode == 0 else None
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout getting Railway logs")
            return None
        except Exception as e:
            print(f"❌ Error getting Railway logs: {e}")
            return None
    
    def analyze_logs(self, logs: str) -> List[Dict]:
        """Analyze logs for error patterns."""
        errors = []
        
        for error_type, pattern in self.error_patterns.items():
            matches = re.findall(pattern, logs, re.IGNORECASE)
            for match in matches:
                error_id = f"{error_type}_{hash(match)}"
                if error_id not in self.fixes_applied:
                    errors.append({
                        'type': error_type,
                        'message': match,
                        'error_id': error_id,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return errors
    
    def suggest_fixes(self, errors: List[Dict]) -> List[str]:
        """Suggest fixes for detected errors."""
        fixes = []
        
        for error in errors:
            error_type = error['type']
            message = error['message']
            
            if error_type == 'module_not_found':
                fixes.append(f"Add missing dependency: pip install {message}")
                fixes.append(f"Or add to requirements.txt: {message}")
                
            elif error_type == 'attribute_error':
                fixes.append(f"Fix attribute error: {message}")
                fixes.append("Check class/method definitions and imports")
                
            elif error_type == 'type_error':
                fixes.append(f"Fix type error: {message}")
                fixes.append("Check function parameters and types")
                
            elif error_type == 'redis_error':
                fixes.append("Check Redis connection configuration")
                fixes.append("Verify REDIS_URL environment variable")
                
            elif error_type == 'startup_failed':
                fixes.append("Check application startup sequence")
                fixes.append("Review lifespan function and dependencies")
                
            elif error_type == 'build_failed':
                fixes.append("Check Dockerfile and build configuration")
                fixes.append("Review requirements.txt for dependency conflicts")
                
        return fixes
    
    def monitor_continuously(self, interval: int = 30):
        """Monitor Railway logs continuously."""
        print("🔍 Starting Railway monitoring...")
        print(f"   Checking logs every {interval} seconds")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                logs = self.get_railway_logs()
                
                if logs:
                    errors = self.analyze_logs(logs)
                    
                    if errors:
                        print(f"\n🚨 Found {len(errors)} errors:")
                        for error in errors:
                            print(f"   • {error['type']}: {error['message']}")
                        
                        fixes = self.suggest_fixes(errors)
                        if fixes:
                            print(f"\n💡 Suggested fixes:")
                            for fix in fixes:
                                print(f"   • {fix}")
                        
                        # Mark errors as seen
                        for error in errors:
                            self.fixes_applied.add(error['error_id'])
                    else:
                        print("✅ No new errors detected")
                else:
                    print("⚠️  Could not retrieve logs")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def check_deployment_status(self) -> Dict:
        """Check current deployment status."""
        try:
            # Try to get service status
            result = subprocess.run(
                ['railway', 'status'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    'status': 'connected',
                    'output': result.stdout
                }
            else:
                return {
                    'status': 'error',
                    'output': result.stderr
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'output': str(e)
            }
    
    def trigger_deployment(self) -> bool:
        """Trigger a new deployment."""
        try:
            print("🚀 Triggering new deployment...")
            result = subprocess.run(
                ['railway', 'up', '--detach'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ Deployment triggered successfully")
                return True
            else:
                print(f"❌ Deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error triggering deployment: {e}")
            return False


def main():
    """Main monitoring function."""
    monitor = RailwayMonitor()
    
    # Check deployment status first
    print("🔍 Checking Railway deployment status...")
    status = monitor.check_deployment_status()
    print(f"Status: {status['status']}")
    print(f"Output: {status['output']}")
    
    # Get current logs
    print("\n📋 Getting current logs...")
    logs = monitor.get_railway_logs()
    
    if logs:
        print(f"Got {len(logs)} characters of logs")
        errors = monitor.analyze_logs(logs)
        
        if errors:
            print(f"\n🚨 Found {len(errors)} errors:")
            for error in errors:
                print(f"   • {error['type']}: {error['message']}")
            
            fixes = monitor.suggest_fixes(errors)
            if fixes:
                print(f"\n💡 Suggested fixes:")
                for fix in fixes:
                    print(f"   • {fix}")
        else:
            print("✅ No errors detected in current logs")
    else:
        print("⚠️  Could not retrieve logs")
    
    # Ask if user wants continuous monitoring
    try:
        response = input("\n🔄 Start continuous monitoring? (y/n): ")
        if response.lower() in ['y', 'yes']:
            monitor.monitor_continuously()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main() 