#!/usr/bin/env python3
"""
SSL/TLS Certificate Management for VivaranAI Production

This script handles SSL certificate generation, renewal, and configuration
for production deployment with multiple certificate authorities.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

class SSLCertificateManager:
    """SSL Certificate management for production deployment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.ssl_dir = self.project_root / "deployment" / "ssl"
        self.certs_dir = self.ssl_dir / "certificates"
        self.keys_dir = self.ssl_dir / "private"
        self.config_dir = self.ssl_dir / "config"
        
        # Create directories if they don't exist
        for directory in [self.certs_dir, self.keys_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
    
    def generate_self_signed_cert(self, domain: str, validity_days: int = 365) -> bool:
        """Generate self-signed certificate for development/testing"""
        logger.info(f"🔐 Generating self-signed certificate for {domain}")
        
        cert_path = self.certs_dir / f"{domain}.crt"
        key_path = self.keys_dir / f"{domain}.key"
        
        # Create OpenSSL configuration
        config_content = f"""
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = California
L = San Francisco
O = VivaranAI
OU = IT Department
CN = {domain}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = {domain}
DNS.2 = *.{domain}
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
"""
        
        config_path = self.config_dir / f"{domain}.conf"
        with open(config_path, "w") as f:
            f.write(config_content)
        
        # Generate private key
        key_cmd = [
            "openssl", "genrsa",
            "-out", str(key_path),
            "2048"
        ]
        
        try:
            subprocess.run(key_cmd, check=True, capture_output=True)
            os.chmod(key_path, 0o600)
            logger.info(f"✅ Generated private key: {key_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate private key: {e}")
            return False
        
        # Generate certificate
        cert_cmd = [
            "openssl", "req",
            "-new", "-x509",
            "-key", str(key_path),
            "-out", str(cert_path),
            "-days", str(validity_days),
            "-config", str(config_path),
            "-extensions", "v3_req"
        ]
        
        try:
            subprocess.run(cert_cmd, check=True, capture_output=True)
            os.chmod(cert_path, 0o644)
            logger.info(f"✅ Generated certificate: {cert_path}")
            
            # Verify certificate
            self.verify_certificate(domain)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate certificate: {e}")
            return False
    
    def setup_letsencrypt(self, domains: List[str], email: str, staging: bool = False) -> bool:
        """Setup Let's Encrypt certificates using Certbot"""
        logger.info(f"🔐 Setting up Let's Encrypt certificates for {domains}")
        
        if not self._check_certbot_installed():
            logger.error("❌ Certbot not installed. Please install certbot first.")
            return False
        
        # Prepare certbot command
        cmd = [
            "certbot", "certonly",
            "--webroot",
            "--webroot-path", "/var/www/html",
            "--email", email,
            "--agree-tos",
            "--non-interactive",
            "--expand"
        ]
        
        if staging:
            cmd.append("--staging")
        
        # Add domains
        for domain in domains:
            cmd.extend(["-d", domain])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Let's Encrypt certificates obtained successfully")
            logger.debug(f"Certbot output: {result.stdout}")
            
            # Copy certificates to our directory
            self._copy_letsencrypt_certs(domains[0])
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to obtain Let's Encrypt certificates: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def setup_cloudflare_origin_cert(self, domain: str, api_token: str) -> bool:
        """Setup Cloudflare Origin Certificate"""
        logger.info(f"🔐 Setting up Cloudflare Origin Certificate for {domain}")
        
        try:
            # Use Cloudflare API to create origin certificate
            import requests
            
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            # Generate CSR first
            csr_path = self.keys_dir / f"{domain}_origin.csr"
            key_path = self.keys_dir / f"{domain}_origin.key"
            
            if not self._generate_csr(domain, str(csr_path), str(key_path)):
                return False
            
            # Read CSR
            with open(csr_path, "r") as f:
                csr_content = f.read()
            
            # Request certificate from Cloudflare
            data = {
                "hostnames": [domain, f"*.{domain}"],
                "requested_validity": 365,
                "request_type": "origin-rsa",
                "csr": csr_content
            }
            
            response = requests.post(
                "https://api.cloudflare.com/client/v4/certificates",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                cert_data = response.json()
                cert_content = cert_data["result"]["certificate"]
                
                cert_path = self.certs_dir / f"{domain}_origin.crt"
                with open(cert_path, "w") as f:
                    f.write(cert_content)
                
                os.chmod(cert_path, 0o644)
                logger.info(f"✅ Cloudflare Origin Certificate saved: {cert_path}")
                return True
            else:
                logger.error(f"❌ Failed to obtain Cloudflare certificate: {response.text}")
                return False
                
        except ImportError:
            logger.error("❌ requests library not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"❌ Error setting up Cloudflare certificate: {e}")
            return False
    
    def renew_certificates(self) -> bool:
        """Renew all certificates that are close to expiry"""
        logger.info("🔄 Checking for certificates to renew...")
        
        renewed = 0
        failed = 0
        
        # Check Let's Encrypt certificates
        try:
            result = subprocess.run(["certbot", "renew", "--dry-run"], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Let's Encrypt certificates are up to date")
            else:
                logger.warning("⚠️  Some Let's Encrypt certificates need renewal")
                # Actual renewal
                subprocess.run(["certbot", "renew"], check=True)
                renewed += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to renew Let's Encrypt certificates: {e}")
            failed += 1
        except FileNotFoundError:
            logger.info("ℹ️  Certbot not found, skipping Let's Encrypt renewal")
        
        # Check self-signed certificates
        for cert_file in self.certs_dir.glob("*.crt"):
            if self._certificate_needs_renewal(cert_file):
                domain = cert_file.stem
                logger.info(f"🔄 Renewing certificate for {domain}")
                if self.generate_self_signed_cert(domain):
                    renewed += 1
                else:
                    failed += 1
        
        logger.info(f"📊 Certificate renewal summary: {renewed} renewed, {failed} failed")
        return failed == 0
    
    def verify_certificate(self, domain: str) -> bool:
        """Verify certificate validity and configuration"""
        cert_path = self.certs_dir / f"{domain}.crt"
        
        if not cert_path.exists():
            logger.error(f"❌ Certificate not found: {cert_path}")
            return False
        
        try:
            # Check certificate details
            cmd = ["openssl", "x509", "-in", str(cert_path), "-text", "-noout"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check expiry
            cmd = ["openssl", "x509", "-in", str(cert_path), "-enddate", "-noout"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            expiry_line = result.stdout.strip()
            expiry_str = expiry_line.replace("notAfter=", "")
            
            # Parse expiry date
            try:
                expiry_date = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
                days_until_expiry = (expiry_date - datetime.now()).days
                
                if days_until_expiry <= 0:
                    logger.error(f"❌ Certificate for {domain} has expired!")
                    return False
                elif days_until_expiry <= 30:
                    logger.warning(f"⚠️  Certificate for {domain} expires in {days_until_expiry} days")
                else:
                    logger.info(f"✅ Certificate for {domain} is valid for {days_until_expiry} days")
                
                return True
                
            except ValueError as e:
                logger.error(f"❌ Failed to parse certificate expiry date: {e}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to verify certificate: {e}")
            return False
    
    def generate_nginx_ssl_config(self, domain: str, cert_type: str = "self-signed") -> str:
        """Generate Nginx SSL configuration"""
        if cert_type == "letsencrypt":
            cert_path = f"/etc/letsencrypt/live/{domain}/fullchain.pem"
            key_path = f"/etc/letsencrypt/live/{domain}/privkey.pem"
        elif cert_type == "cloudflare":
            cert_path = f"/etc/ssl/certs/{domain}_origin.crt"
            key_path = f"/etc/ssl/private/{domain}_origin.key"
        else:  # self-signed
            cert_path = f"/etc/ssl/certs/{domain}.crt"
            key_path = f"/etc/ssl/private/{domain}.key"
        
        config = f"""
# SSL Configuration for {domain}
server {{
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name {domain};

    # SSL Certificate Configuration
    ssl_certificate {cert_path};
    ssl_certificate_key {key_path};

    # SSL Security Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA256:DHE-RSA-AES256-SHA;
    ssl_prefer_server_ciphers off;

    # SSL Session Configuration
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Application Configuration
    location / {{
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Ssl on;
        
        # Timeout configuration
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}

    # Health check endpoint
    location /health {{
        proxy_pass http://127.0.0.1:8001/health;
        access_log off;
    }}

    # Static files (if any)
    location /static/ {{
        alias /var/www/vivaranai/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}
}}

# HTTP to HTTPS redirect
server {{
    listen 80;
    listen [::]:80;
    server_name {domain};
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {{
        root /var/www/html;
    }}
    
    # Redirect all other traffic to HTTPS
    location / {{
        return 301 https://$server_name$request_uri;
    }}
}}
"""
        return config
    
    def _check_certbot_installed(self) -> bool:
        """Check if certbot is installed"""
        try:
            subprocess.run(["certbot", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _copy_letsencrypt_certs(self, domain: str):
        """Copy Let's Encrypt certificates to our directory"""
        le_cert_path = f"/etc/letsencrypt/live/{domain}/fullchain.pem"
        le_key_path = f"/etc/letsencrypt/live/{domain}/privkey.pem"
        
        our_cert_path = self.certs_dir / f"{domain}_letsencrypt.crt"
        our_key_path = self.keys_dir / f"{domain}_letsencrypt.key"
        
        try:
            subprocess.run(["cp", le_cert_path, str(our_cert_path)], check=True)
            subprocess.run(["cp", le_key_path, str(our_key_path)], check=True)
            
            os.chmod(our_cert_path, 0o644)
            os.chmod(our_key_path, 0o600)
            
            logger.info(f"✅ Copied Let's Encrypt certificates for {domain}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to copy Let's Encrypt certificates: {e}")
    
    def _generate_csr(self, domain: str, csr_path: str, key_path: str) -> bool:
        """Generate Certificate Signing Request"""
        # Generate private key
        key_cmd = ["openssl", "genrsa", "-out", key_path, "2048"]
        
        try:
            subprocess.run(key_cmd, check=True, capture_output=True)
            os.chmod(key_path, 0o600)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate private key: {e}")
            return False
        
        # Generate CSR
        csr_cmd = [
            "openssl", "req", "-new",
            "-key", key_path,
            "-out", csr_path,
            "-subj", f"/C=US/ST=CA/L=SF/O=VivaranAI/CN={domain}"
        ]
        
        try:
            subprocess.run(csr_cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate CSR: {e}")
            return False
    
    def _certificate_needs_renewal(self, cert_path: Path) -> bool:
        """Check if certificate needs renewal (within 30 days)"""
        try:
            cmd = ["openssl", "x509", "-in", str(cert_path), "-enddate", "-noout"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            expiry_line = result.stdout.strip()
            expiry_str = expiry_line.replace("notAfter=", "")
            
            expiry_date = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
            days_until_expiry = (expiry_date - datetime.now()).days
            
            return days_until_expiry <= 30
            
        except (subprocess.CalledProcessError, ValueError):
            return True  # Renew if we can't determine expiry

def main():
    """CLI interface for SSL certificate management"""
    parser = argparse.ArgumentParser(description="VivaranAI SSL Certificate Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Self-signed certificate
    self_signed_parser = subparsers.add_parser("self-signed", help="Generate self-signed certificate")
    self_signed_parser.add_argument("domain", help="Domain name")
    self_signed_parser.add_argument("--days", type=int, default=365, help="Validity in days")
    
    # Let's Encrypt
    letsencrypt_parser = subparsers.add_parser("letsencrypt", help="Setup Let's Encrypt certificate")
    letsencrypt_parser.add_argument("domains", nargs="+", help="Domain names")
    letsencrypt_parser.add_argument("--email", required=True, help="Email for Let's Encrypt")
    letsencrypt_parser.add_argument("--staging", action="store_true", help="Use staging environment")
    
    # Cloudflare Origin
    cloudflare_parser = subparsers.add_parser("cloudflare", help="Setup Cloudflare Origin certificate")
    cloudflare_parser.add_argument("domain", help="Domain name")
    cloudflare_parser.add_argument("--token", required=True, help="Cloudflare API token")
    
    # Renew certificates
    subparsers.add_parser("renew", help="Renew certificates")
    
    # Verify certificate
    verify_parser = subparsers.add_parser("verify", help="Verify certificate")
    verify_parser.add_argument("domain", help="Domain name")
    
    # Generate Nginx config
    nginx_parser = subparsers.add_parser("nginx-config", help="Generate Nginx SSL configuration")
    nginx_parser.add_argument("domain", help="Domain name")
    nginx_parser.add_argument("--type", choices=["self-signed", "letsencrypt", "cloudflare"], 
                            default="self-signed", help="Certificate type")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    ssl_mgr = SSLCertificateManager()
    
    try:
        if args.command == "self-signed":
            success = ssl_mgr.generate_self_signed_cert(args.domain, args.days)
            return 0 if success else 1
            
        elif args.command == "letsencrypt":
            success = ssl_mgr.setup_letsencrypt(args.domains, args.email, args.staging)
            return 0 if success else 1
            
        elif args.command == "cloudflare":
            success = ssl_mgr.setup_cloudflare_origin_cert(args.domain, args.token)
            return 0 if success else 1
            
        elif args.command == "renew":
            success = ssl_mgr.renew_certificates()
            return 0 if success else 1
            
        elif args.command == "verify":
            success = ssl_mgr.verify_certificate(args.domain)
            return 0 if success else 1
            
        elif args.command == "nginx-config":
            config = ssl_mgr.generate_nginx_ssl_config(args.domain, args.type)
            print(config)
            return 0
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 