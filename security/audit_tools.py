#!/usr/bin/env python3
"""
Security Audit Tools for VivaranAI Production

This module provides comprehensive security auditing, vulnerability assessment,
and penetration testing tools for production security validation.
"""

import asyncio
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import socket
import requests
from urllib.parse import urlparse
import concurrent.futures
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class SeverityLevel(Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a security issue found during audit"""
    id: str
    title: str
    severity: SeverityLevel
    description: str
    location: str
    evidence: str = ""
    recommendation: str = ""
    cve_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'severity': self.severity.value,
            'description': self.description,
            'location': self.location,
            'evidence': self.evidence,
            'recommendation': self.recommendation,
            'cve_id': self.cve_id,
            'references': self.references,
            'timestamp': self.timestamp.isoformat()
        }


class SecurityAuditReport:
    """Comprehensive security audit report"""
    
    def __init__(self):
        self.scan_start_time = datetime.now()
        self.scan_end_time = None
        self.issues: List[SecurityIssue] = []
        self.scan_summary = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
    
    def add_issue(self, issue: SecurityIssue):
        """Add security issue to report"""
        self.issues.append(issue)
        self.scan_summary[issue.severity.value] += 1
        
        logger.warning(f"🔒 Security issue found: {issue.title}", 
                      severity=issue.severity.value,
                      location=issue.location)
    
    def finalize_report(self):
        """Finalize the security audit report"""
        self.scan_end_time = datetime.now()
        scan_duration = (self.scan_end_time - self.scan_start_time).total_seconds()
        
        total_issues = sum(self.scan_summary.values())
        critical_high = self.scan_summary['critical'] + self.scan_summary['high']
        
        logger.info(f"📊 Security audit completed in {scan_duration:.2f}s",
                   total_issues=total_issues,
                   critical_high=critical_high,
                   summary=self.scan_summary)
    
    def get_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        weights = {
            'critical': 25,
            'high': 15,
            'medium': 8,
            'low': 3,
            'info': 1
        }
        
        total_score = sum(self.scan_summary[severity] * weight 
                         for severity, weight in weights.items())
        
        # Normalize to 0-100 scale
        return min(total_score, 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'scan_start_time': self.scan_start_time.isoformat(),
            'scan_end_time': self.scan_end_time.isoformat() if self.scan_end_time else None,
            'scan_duration_seconds': (self.scan_end_time - self.scan_start_time).total_seconds() if self.scan_end_time else None,
            'summary': self.scan_summary,
            'risk_score': self.get_risk_score(),
            'total_issues': sum(self.scan_summary.values()),
            'issues': [issue.to_dict() for issue in self.issues]
        }


class CodeSecurityScanner:
    """Static code security analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = SecurityAuditReport()
        
        # Common security patterns
        self.security_patterns = {
            'hardcoded_secrets': [
                r'(?i)(password|passwd|pwd|secret|token|key|api[_-]?key)\s*[=:]\s*["\']([^"\']{8,})["\']',
                r'(?i)(password|passwd|pwd|secret|token|key|api[_-]?key)\s*[=:]\s*([a-zA-Z0-9+/]{20,})',
                r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
                r'sk-[a-zA-Z0-9]{32,}',  # OpenAI API keys
                r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
                r'ghs_[a-zA-Z0-9]{36}',  # GitHub app tokens
            ],
            'sql_injection': [
                r'(?i)execute\s*\(\s*["\'].*%s.*["\']',
                r'(?i)cursor\.execute\s*\(\s*["\'].*%s.*["\']',
                r'(?i)query\s*\(\s*["\'].*%s.*["\']',
                r'(?i)\.format\s*\(\s*.*\)\s*["\']',
            ],
            'command_injection': [
                r'(?i)os\.system\s*\(\s*["\'].*%s.*["\']',
                r'(?i)subprocess\.call\s*\(\s*["\'].*%s.*["\']',
                r'(?i)subprocess\.run\s*\(\s*["\'].*%s.*["\']',
                r'(?i)subprocess\.Popen\s*\(\s*["\'].*%s.*["\']',
            ],
            'path_traversal': [
                r'(?i)open\s*\(\s*.*\+.*["\']',
                r'(?i)\.\./',
                r'(?i)%2e%2e%2f',
            ],
            'insecure_crypto': [
                r'(?i)md5\s*\(',
                r'(?i)sha1\s*\(',
                r'(?i)DES\s*\(',
                r'(?i)RC4\s*\(',
                r'(?i)ssl_verify\s*=\s*False',
                r'(?i)verify\s*=\s*False',
            ],
            'debug_code': [
                r'(?i)print\s*\(\s*["\'].*password.*["\']',
                r'(?i)print\s*\(\s*["\'].*secret.*["\']',
                r'(?i)print\s*\(\s*["\'].*token.*["\']',
                r'(?i)debug\s*=\s*True',
            ]
        }
    
    def scan_project(self) -> SecurityAuditReport:
        """Scan entire project for security issues"""
        logger.info("🔍 Starting code security scan...")
        
        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        # Scan each file
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self._scan_file_content(file_path, content)
            except Exception as e:
                logger.warning(f"⚠️  Could not scan {file_path}: {e}")
        
        # Scan configuration files
        self._scan_config_files()
        
        # Check dependencies
        self._check_dependencies()
        
        self.report.finalize_report()
        return self.report
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '/.git/',
            '/.venv/',
            '/venv/',
            '/node_modules/',
            '__pycache__',
            '.pyc',
            '/migrations/',
            '/test_',
            '/tests/',
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _scan_file_content(self, file_path: Path, content: str):
        """Scan file content for security issues"""
        lines = content.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        self._create_security_issue(
                            category, file_path, line_num, line, pattern
                        )
    
    def _create_security_issue(self, category: str, file_path: Path, 
                              line_num: int, line: str, pattern: str):
        """Create security issue from pattern match"""
        issue_id = hashlib.md5(f"{category}:{file_path}:{line_num}".encode()).hexdigest()[:8]
        
        severity_map = {
            'hardcoded_secrets': SeverityLevel.CRITICAL,
            'sql_injection': SeverityLevel.HIGH,
            'command_injection': SeverityLevel.HIGH,
            'path_traversal': SeverityLevel.HIGH,
            'insecure_crypto': SeverityLevel.MEDIUM,
            'debug_code': SeverityLevel.LOW
        }
        
        recommendations = {
            'hardcoded_secrets': "Use environment variables or secret management systems",
            'sql_injection': "Use parameterized queries or ORM",
            'command_injection': "Sanitize inputs and use shell=False",
            'path_traversal': "Validate and sanitize file paths",
            'insecure_crypto': "Use secure algorithms like SHA-256 or AES",
            'debug_code': "Remove debug code from production"
        }
        
        issue = SecurityIssue(
            id=issue_id,
            title=f"{category.replace('_', ' ').title()} Vulnerability",
            severity=severity_map.get(category, SeverityLevel.MEDIUM),
            description=f"Potential {category} vulnerability detected",
            location=f"{file_path}:{line_num}",
            evidence=line.strip(),
            recommendation=recommendations.get(category, "Review and fix the identified issue")
        )
        
        self.report.add_issue(issue)
    
    def _scan_config_files(self):
        """Scan configuration files for security issues"""
        config_files = [
            '.env', '.env.local', '.env.production',
            'config.yaml', 'config.yml', 'settings.yaml',
            'docker-compose.yml', 'Dockerfile'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        
                    # Check for hardcoded secrets
                    if re.search(r'(?i)(password|secret|token|key)\s*[=:]\s*["\']?[a-zA-Z0-9+/]{8,}', content):
                        issue = SecurityIssue(
                            id=f"config_{config_file}",
                            title="Hardcoded Secrets in Configuration",
                            severity=SeverityLevel.HIGH,
                            description="Configuration file contains hardcoded secrets",
                            location=str(config_path),
                            evidence="Hardcoded credentials found",
                            recommendation="Use environment variables or secret management"
                        )
                        self.report.add_issue(issue)
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not scan config {config_path}: {e}")
    
    def _check_dependencies(self):
        """Check for vulnerable dependencies"""
        requirements_files = [
            'requirements.txt', 'requirements-dev.txt', 'pyproject.toml'
        ]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    # Run safety check if available
                    result = subprocess.run(
                        ['safety', 'check', '-r', str(req_path)],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode != 0 and "vulnerabilities found" in result.stdout:
                        issue = SecurityIssue(
                            id=f"deps_{req_file}",
                            title="Vulnerable Dependencies",
                            severity=SeverityLevel.HIGH,
                            description="Vulnerable dependencies found",
                            location=str(req_path),
                            evidence=result.stdout[:500],
                            recommendation="Update vulnerable dependencies"
                        )
                        self.report.add_issue(issue)
                        
                except FileNotFoundError:
                    logger.info("ℹ️  'safety' tool not found, skipping dependency check")
                except Exception as e:
                    logger.warning(f"⚠️  Could not check dependencies: {e}")


class NetworkSecurityScanner:
    """Network security and web application scanner"""
    
    def __init__(self, target_host: str, target_port: int = 8001):
        self.target_host = target_host
        self.target_port = target_port
        self.report = SecurityAuditReport()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VivaranAI-Security-Scanner/1.0'
        })
    
    def scan_network(self) -> SecurityAuditReport:
        """Perform network security scan"""
        logger.info(f"🌐 Starting network security scan for {self.target_host}:{self.target_port}")
        
        # Port scanning
        self._scan_ports()
        
        # SSL/TLS testing
        self._test_ssl_tls()
        
        # HTTP security headers
        self._check_security_headers()
        
        # Web application testing
        self._test_web_application()
        
        # Authentication testing
        self._test_authentication()
        
        self.report.finalize_report()
        return self.report
    
    def _scan_ports(self):
        """Scan common ports for open services"""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 8000, 8001, 8080, 8443]
        
        logger.info(f"🔍 Scanning ports on {self.target_host}")
        
        open_ports = []
        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex((self.target_host, port))
                if result == 0:
                    open_ports.append(port)
            except Exception:
                pass
            finally:
                sock.close()
        
        # Check for unnecessary open ports
        expected_ports = [80, 443, self.target_port]
        unexpected_ports = [port for port in open_ports if port not in expected_ports]
        
        if unexpected_ports:
            issue = SecurityIssue(
                id="open_ports",
                title="Unnecessary Open Ports",
                severity=SeverityLevel.MEDIUM,
                description="Unexpected ports are open",
                location=f"{self.target_host}",
                evidence=f"Open ports: {', '.join(map(str, unexpected_ports))}",
                recommendation="Close unnecessary ports and services"
            )
            self.report.add_issue(issue)
    
    def _test_ssl_tls(self):
        """Test SSL/TLS configuration"""
        if self.target_port not in [443, 8443]:
            return
        
        logger.info(f"🔒 Testing SSL/TLS configuration")
        
        try:
            # Test SSL connection
            context = ssl.create_default_context()
            
            with socket.create_connection((self.target_host, self.target_port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.target_host) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    # Check certificate validity
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.now()).days
                    
                    if days_until_expiry < 30:
                        issue = SecurityIssue(
                            id="ssl_cert_expiry",
                            title="SSL Certificate Expiring Soon",
                            severity=SeverityLevel.HIGH,
                            description="SSL certificate expires within 30 days",
                            location=f"{self.target_host}:{self.target_port}",
                            evidence=f"Expires on {cert['notAfter']}",
                            recommendation="Renew SSL certificate"
                        )
                        self.report.add_issue(issue)
                    
                    # Check cipher strength
                    if cipher and cipher[1] in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.1']:
                        issue = SecurityIssue(
                            id="weak_ssl_version",
                            title="Weak SSL/TLS Version",
                            severity=SeverityLevel.HIGH,
                            description="Server supports weak SSL/TLS versions",
                            location=f"{self.target_host}:{self.target_port}",
                            evidence=f"Version: {cipher[1]}",
                            recommendation="Disable weak SSL/TLS versions"
                        )
                        self.report.add_issue(issue)
                        
        except Exception as e:
            logger.warning(f"⚠️  SSL/TLS test failed: {e}")
    
    def _check_security_headers(self):
        """Check HTTP security headers"""
        try:
            url = f"http://{self.target_host}:{self.target_port}"
            response = self.session.get(url, timeout=10)
            
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,  # Should be present
                'Content-Security-Policy': None,    # Should be present
                'Referrer-Policy': None,           # Should be present
            }
            
            for header, expected_value in required_headers.items():
                if header not in response.headers:
                    issue = SecurityIssue(
                        id=f"missing_{header.lower().replace('-', '_')}",
                        title=f"Missing {header} Header",
                        severity=SeverityLevel.MEDIUM,
                        description=f"Security header {header} is missing",
                        location=url,
                        evidence=f"Header '{header}' not found in response",
                        recommendation=f"Add {header} security header"
                    )
                    self.report.add_issue(issue)
                
                elif expected_value and response.headers.get(header) not in expected_value:
                    issue = SecurityIssue(
                        id=f"weak_{header.lower().replace('-', '_')}",
                        title=f"Weak {header} Header",
                        severity=SeverityLevel.LOW,
                        description=f"Security header {header} has weak value",
                        location=url,
                        evidence=f"Header value: {response.headers.get(header)}",
                        recommendation=f"Strengthen {header} header configuration"
                    )
                    self.report.add_issue(issue)
                    
        except Exception as e:
            logger.warning(f"⚠️  Security headers check failed: {e}")
    
    def _test_web_application(self):
        """Test web application for common vulnerabilities"""
        base_url = f"http://{self.target_host}:{self.target_port}"
        
        # Test for directory traversal
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in traversal_payloads:
            try:
                response = self.session.get(f"{base_url}/{payload}", timeout=5)
                if "root:" in response.text or "localhost" in response.text:
                    issue = SecurityIssue(
                        id="directory_traversal",
                        title="Directory Traversal Vulnerability",
                        severity=SeverityLevel.HIGH,
                        description="Application is vulnerable to directory traversal",
                        location=f"{base_url}/{payload}",
                        evidence=response.text[:200],
                        recommendation="Implement proper input validation and path sanitization"
                    )
                    self.report.add_issue(issue)
                    break
            except Exception:
                continue
        
        # Test for SQL injection (basic test)
        sql_payloads = ["'", "' OR '1'='1", "'; DROP TABLE users; --"]
        
        for payload in sql_payloads:
            try:
                response = self.session.get(f"{base_url}/api/search?q={payload}", timeout=5)
                if "sql" in response.text.lower() or "mysql" in response.text.lower():
                    issue = SecurityIssue(
                        id="sql_injection",
                        title="Potential SQL Injection",
                        severity=SeverityLevel.HIGH,
                        description="Application may be vulnerable to SQL injection",
                        location=f"{base_url}/api/search",
                        evidence=response.text[:200],
                        recommendation="Use parameterized queries and input validation"
                    )
                    self.report.add_issue(issue)
                    break
            except Exception:
                continue
    
    def _test_authentication(self):
        """Test authentication mechanisms"""
        base_url = f"http://{self.target_host}:{self.target_port}"
        
        # Test for common credentials
        common_creds = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("user", "user"),
            ("test", "test")
        ]
        
        for username, password in common_creds:
            try:
                response = self.session.post(
                    f"{base_url}/api/login",
                    json={"username": username, "password": password},
                    timeout=5
                )
                
                if response.status_code == 200:
                    issue = SecurityIssue(
                        id="weak_credentials",
                        title="Weak Default Credentials",
                        severity=SeverityLevel.HIGH,
                        description="Application accepts weak default credentials",
                        location=f"{base_url}/api/login",
                        evidence=f"Login successful with {username}:{password}",
                        recommendation="Change default credentials and enforce strong password policy"
                    )
                    self.report.add_issue(issue)
                    break
                    
            except Exception:
                continue


class SecurityAuditor:
    """Main security auditor class"""
    
    def __init__(self, project_root: Path, target_host: str = "localhost", target_port: int = 8001):
        self.project_root = project_root
        self.target_host = target_host
        self.target_port = target_port
        self.audit_start_time = datetime.now()
        
    def run_full_audit(self) -> Dict[str, SecurityAuditReport]:
        """Run complete security audit"""
        logger.info("🔒 Starting comprehensive security audit...")
        
        results = {}
        
        # Code security scan
        logger.info("📁 Running code security scan...")
        code_scanner = CodeSecurityScanner(self.project_root)
        results['code_security'] = code_scanner.scan_project()
        
        # Network security scan
        logger.info("🌐 Running network security scan...")
        network_scanner = NetworkSecurityScanner(self.target_host, self.target_port)
        results['network_security'] = network_scanner.scan_network()
        
        # Generate combined report
        self._generate_combined_report(results)
        
        return results
    
    def _generate_combined_report(self, results: Dict[str, SecurityAuditReport]):
        """Generate combined security audit report"""
        audit_end_time = datetime.now()
        audit_duration = (audit_end_time - self.audit_start_time).total_seconds()
        
        # Combine all issues
        all_issues = []
        for report in results.values():
            all_issues.extend(report.issues)
        
        # Calculate overall risk
        total_critical = sum(report.scan_summary['critical'] for report in results.values())
        total_high = sum(report.scan_summary['high'] for report in results.values())
        total_issues = len(all_issues)
        
        overall_risk = "LOW"
        if total_critical > 0:
            overall_risk = "CRITICAL"
        elif total_high > 0:
            overall_risk = "HIGH"
        elif total_issues > 5:
            overall_risk = "MEDIUM"
        
        # Generate report
        report_data = {
            'audit_metadata': {
                'start_time': self.audit_start_time.isoformat(),
                'end_time': audit_end_time.isoformat(),
                'duration_seconds': audit_duration,
                'target_host': self.target_host,
                'target_port': self.target_port,
                'project_root': str(self.project_root)
            },
            'overall_summary': {
                'total_issues': total_issues,
                'critical': total_critical,
                'high': total_high,
                'overall_risk': overall_risk,
                'scan_types': list(results.keys())
            },
            'detailed_results': {
                scan_type: report.to_dict() 
                for scan_type, report in results.items()
            }
        }
        
        # Save report
        report_path = self.project_root / "security_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📊 Security audit completed in {audit_duration:.2f}s",
                   total_issues=total_issues,
                   critical=total_critical,
                   high=total_high,
                   overall_risk=overall_risk,
                   report_path=str(report_path))
        
        # Print summary
        print("\n" + "="*60)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("="*60)
        print(f"Overall Risk Level: {overall_risk}")
        print(f"Total Issues Found: {total_issues}")
        print(f"Critical Issues: {total_critical}")
        print(f"High Issues: {total_high}")
        print(f"Report saved to: {report_path}")
        
        if total_critical > 0:
            print("\n⚠️  CRITICAL ISSUES FOUND! Immediate action required.")
        elif total_high > 0:
            print("\n⚠️  HIGH PRIORITY ISSUES FOUND! Action required.")
        else:
            print("\n✅ No critical or high priority issues found.")
        
        print("="*60)


def main():
    """CLI interface for security audit"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VivaranAI Security Audit Tool")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--target-host", default="localhost", help="Target host for network scan")
    parser.add_argument("--target-port", type=int, default=8001, help="Target port for network scan")
    parser.add_argument("--code-only", action="store_true", help="Run only code security scan")
    parser.add_argument("--network-only", action="store_true", help="Run only network security scan")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    auditor = SecurityAuditor(project_root, args.target_host, args.target_port)
    
    try:
        if args.code_only:
            scanner = CodeSecurityScanner(project_root)
            results = {'code_security': scanner.scan_project()}
        elif args.network_only:
            scanner = NetworkSecurityScanner(args.target_host, args.target_port)
            results = {'network_security': scanner.scan_network()}
        else:
            results = auditor.run_full_audit()
        
        return 0 if all(report.get_risk_score() < 50 for report in results.values()) else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Security audit cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Security audit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 