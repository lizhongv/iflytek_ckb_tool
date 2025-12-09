# -*- coding: utf-8 -*-

"""
Login module for UAP authentication
Handles the complete login flow including configuration, public key retrieval,
tenant information, UAP login, redirect, and session management
"""
import asyncio
import base64
import hashlib
import json
import re
import ssl
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from typing import Optional
import sys
import os

# Add project root to path for importing conf modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from conf.error_codes import ErrorCode
from spark_api_tool.config import config_manager

import logging
logger = logging.getLogger(__name__)


class LoginManager:
    """Manager for UAP authentication and login flow"""
    
    def __init__(self):
        """Initialize LoginManager with configuration"""
        # Load configuration
        self.uap_ip = config_manager.server.uap_ip
        self.uap_port = config_manager.server.uap_port
        self.ckb_ip = config_manager.server.ckb_ip
        self.ckb_port = config_manager.server.ckb_port
        self.login_name = config_manager.server.login_name
        self.password = config_manager.server.password
        
        # Runtime variables
        self.session: Optional[aiohttp.ClientSession] = None
        self.publickey: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.redirect_ip: Optional[str] = None
        self.redirect_port: Optional[str] = None
        self.redirect_path: Optional[str] = None
        self.cookie_session: Optional[str] = None
        self.user_id: Optional[str] = None
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context supporting legacy renegotiation"""
        ssl_context = ssl.create_default_context()
        # Allow unsafe legacy renegotiation (for compatibility with old servers)
        ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT
        return ssl_context
    
    async def __aenter__(self):
        """Async context manager entry"""
        ssl_context = self._create_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def ensure_session(self) -> None:
        """Ensure session is created"""
        if self.session is None:
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
    
    def rsa_encrypt(self, data: str, public_key_str: str) -> Optional[str]:
        """
        RSA encryption
        
        Args:
            data: Data to encrypt
            public_key_str: Public key string (base64 encoded)
            
        Returns:
            Encrypted data (base64 encoded) or None if failed
        """
        try:
            # Clean public key string
            public_key_str = re.sub(r'\s+', '', public_key_str)
            
            # Base64 decode
            key_bytes = base64.b64decode(public_key_str)
            
            # Load public key
            public_key = serialization.load_der_public_key(key_bytes, backend=default_backend())
            
            # Encrypt data
            encrypted = public_key.encrypt(
                data.encode('utf-8'),
                padding.PKCS1v15()
            )
            
            # Base64 encode and return
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            return None
    
    def md5_hash(self, text: str) -> str:
        """
        MD5 hash
        
        Args:
            text: Text to hash
            
        Returns:
            MD5 hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def get_config(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Get configuration
        
        Args:
            intranet: Whether to use intranet address. If None, use value from config
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()
        
        use_intranet = intranet if intranet is not None else config_manager.server.intranet

        if use_intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/getConfig"
        else:
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/corpus_uap_server/getConfig"
        
        try:
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Get config: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    return True, None
                else:
                    logger.error(f"Get config failed: HTTP {response.status}")
                    return False, ErrorCode.AUTH_UAP_FAILED
        except Exception as e:
            logger.error(f"Get config failed: {e}")
            return False, ErrorCode.AUTH_UAP_FAILED
    
    async def get_public_key(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Get public key
        
        Args:
            intranet: Whether to use intranet address
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()

        use_intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if use_intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/api/v2/getPublicKey"
        else:
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/corpus_uap_server/api/v2/getPublicKey"
        
        try:
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.publickey = data.get('data', {}).get('publicKey')
                    if self.publickey:
                        logger.debug(f"Public key: {self.publickey}")
                        return True, None
                    else:
                        logger.error("Public key not found in response")
                        return False, ErrorCode.AUTH_GET_PUBLIC_KEY_FAILED
                else:
                    logger.error(f"Get public key failed: HTTP {response.status}")
                    return False, ErrorCode.AUTH_GET_PUBLIC_KEY_FAILED
        except Exception as e:
            logger.error(f"Get public key failed: {e}")
            return False, ErrorCode.AUTH_GET_PUBLIC_KEY_FAILED
    
    async def get_tenant_info(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Get tenant information
        
        Args:
            intranet: Whether to use intranet address
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()

        use_intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if use_intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/search/tenants/byName"
        else:
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/corpus_uap_server/search/tenants/byName"
        
        # Prepare encrypted data
        password_md5 = self.md5_hash(self.password)
        json_data = {
            "password": password_md5,
            "loginName": self.login_name,
            "tenantId": None,
            "tenantName": None,
            "appCode": "spark_knowledge_base",
            "credentialType": "Password"
        }
        
        json_str = json.dumps(json_data, separators=(',', ':'))
        encrypted_data = self.rsa_encrypt(json_str, self.publickey)
        
        if not encrypted_data:
            logger.error("Encryption failed")
            return False, ErrorCode.AUTH_GET_TENANT_INFO_FAILED
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            async with self.session.post(url, data=encrypted_data, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tenant_data = data.get('data', [])
                    if tenant_data:
                        self.tenant_id = tenant_data[0].get('id')
                        if self.tenant_id:
                            logger.debug(f"Tenant ID retrieved: {self.tenant_id}")
                            return True, None
                        else:
                            logger.error("Tenant ID not found in response")
                            return False, ErrorCode.AUTH_GET_TENANT_INFO_FAILED
                    else:
                        logger.error("Tenant data is empty")
                        return False, ErrorCode.AUTH_GET_TENANT_INFO_FAILED
                else:
                    logger.error(f"Get tenant info failed: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text[:200]}...")
                    return False, ErrorCode.AUTH_GET_TENANT_INFO_FAILED
        except Exception as e:
            logger.error(f"Get tenant info failed: {e}")
            return False, ErrorCode.AUTH_GET_TENANT_INFO_FAILED
    
    async def login_uap(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Login to UAP
        
        Args:
            intranet: Whether to use intranet address. If None, use value from config
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()
        
        use_intranet = intranet if intranet is not None else config_manager.server.intranet

        if use_intranet:
            base_url = f"http://{self.uap_ip}:{self.uap_port}"
        else:
            base_url = config_manager.server.external_base_url
        url = f"{base_url}/corpus_uap_server/api/v2/login"
    
        # Determine service URL and redirect URL based on intranet flag
        if use_intranet:
            service_url = f"http://{self.ckb_ip}:{self.ckb_port}/app-skb/auth/info"
            redirect_url = f"http://{self.ckb_ip}:{self.ckb_port}/app-skb/front/knowledge-space"
        else:
            # Use external_base_url from config
            service_url = f"{base_url}/app-skb/auth/info"
            redirect_url = f"{base_url}/app-skb/front/knowledge-space"
        
        params = {
            'service': service_url,
            'credentialType': 'Password',
            'redirect': redirect_url,
            'rememberMe': 'true',
            'tenantId': self.tenant_id
        }
        
        # Prepare encrypted data
        password_md5 = self.md5_hash(self.password)
        json_data = {
            "loginName": self.login_name,
            "password": password_md5,
            "tenantId": self.tenant_id,
            "tenantName": None,
            "appCode": "spark_knowledge_base",
            "credentialType": "Password"
        }
        
        json_str = json.dumps(json_data, separators=(',', ':'))
        encrypted_data = self.rsa_encrypt(json_str, self.publickey)
        
        if not encrypted_data:
            logger.error("RSA encryption failed for login")
            return False, ErrorCode.AUTH_LOGIN_FAILED

        headers = {
            'Content-Type': 'application/json',
            'Referer': base_url
        }
        
        try:
            # Add random delay
            await asyncio.sleep(1 + (time.time() % 1))
            
            async with self.session.post(url, data=encrypted_data, headers=headers, params=params) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        redirect_url = data.get('data', {}).get('redirectUrl')
                        
                        if redirect_url:
                            # Parse redirect URL
                            parsed = urlparse(redirect_url)
                            self.redirect_ip = parsed.hostname
                            self.redirect_port = str(parsed.port)
                            self.redirect_path = parsed.path
                            if parsed.query:
                                self.redirect_path += '?' + parsed.query
                            logger.debug(f"Redirect URL parsed: {self.redirect_ip}:{self.redirect_port}{self.redirect_path}")
                            return True, None
                        else:
                            logger.error("No redirect URL in response")
                            logger.error(f"Response data: {data}")
                            return False, ErrorCode.AUTH_LOGIN_FAILED
                            
                    except Exception as e:
                        logger.error(f"JSON parsing failed: {e}")
                        response_text = await response.text()
                        logger.error(f"Response body: {response_text[:200]}...")
                        return False, ErrorCode.AUTH_LOGIN_FAILED
                        
                else:
                    logger.error(f"UAP login failed: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text[:200]}...")
                    return False, ErrorCode.AUTH_LOGIN_FAILED
                    
        except Exception as e:
            logger.error(f"UAP login exception: {e}")
            return False, ErrorCode.AUTH_LOGIN_FAILED
    
    async def redirect_login(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Redirect login
        
        Args:
            intranet: Whether to use intranet address
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()

        use_intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if use_intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}{self.redirect_path}"
        else:
            # Use external network address from config
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/corpus_uap_manager/getCsrfToken"
        
        headers = {
            'Content-Type': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'accept-encoding': 'gzip, deflate'
        }
        
        try:
            # Allow automatic redirect following
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                # Check redirect history
                if response.history:
                    for i, hist_resp in enumerate(response.history):
                        logger.debug(f"  {i+1}. {hist_resp.status} -> {hist_resp.url}")
                        # Check cookie in each redirect response
                        set_cookie = hist_resp.headers.get('Set-Cookie', '')
                        if set_cookie:
                            logger.debug(f"    Set-Cookie: {set_cookie}")
                
                if response.status == 200:
                    # Check all response headers for cookies
                    session_found = False
                    
                    # First check final response cookie
                    set_cookie_header = response.headers.get('Set-Cookie', '')
                    
                    if set_cookie_header:
                        # Use JMeter script regex logic
                        cookie_match = re.search(r'([^;]+); Path=', set_cookie_header)
                        if cookie_match:
                            self.cookie_session = cookie_match.group(1)
                            session_found = True
                        else:
                            # Fallback: extract first cookie
                            cookie_match = re.search(r'([^;]+)', set_cookie_header)
                            if cookie_match:
                                self.cookie_session = cookie_match.group(1)
                                session_found = True
                    
                    # If final response has no cookie, check redirect history
                    if not session_found and response.history:
                        for hist_resp in response.history:
                            set_cookie = hist_resp.headers.get('Set-Cookie', '')
                            if set_cookie:
                                # Use JMeter script regex logic
                                cookie_match = re.search(r'([^;]+); Path=', set_cookie)
                                if cookie_match:
                                    self.cookie_session = cookie_match.group(1)
                                    session_found = True
                                    break
                                else:
                                    # Fallback
                                    cookie_match = re.search(r'([^;]+)', set_cookie)
                                    if cookie_match:
                                        self.cookie_session = cookie_match.group(1)
                                        session_found = True
                                        break
                    
                    # If still no cookie found, check session object cookies
                    if not session_found:
                        cookies = self.session.cookie_jar
                        
                        if cookies:
                            # Try to construct cookie string
                            cookie_parts = []
                            for cookie in cookies:
                                cookie_parts.append(f"{cookie.key}={cookie.value}")
                            
                            if cookie_parts:
                                self.cookie_session = "; ".join(cookie_parts)
                                session_found = True
                    
                    if session_found:
                        logger.debug("Session cookie retrieved successfully")
                        return True, None
                    else:
                        logger.error("Failed to get any session cookie")
                        # Show first 500 chars of response body for debugging
                        response_text = await response.text()
                        logger.error(f"Response body (first 500 chars): {response_text[:500]}")
                        return False, ErrorCode.AUTH_REDIRECT_FAILED
                        
                else:
                    logger.error(f"Redirect login failed: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text[:200]}...")
                    return False, ErrorCode.AUTH_REDIRECT_FAILED
                    
        except Exception as e:
            logger.error(f"Redirect login exception: {e}")
            return False, ErrorCode.AUTH_REDIRECT_FAILED
    
    async def access_knowledge_homepage(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Access knowledge base homepage
        
        Args:
            intranet: Whether to use intranet address
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()

        use_intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if use_intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}/app-skb/front/knowledge-space"
        else:
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/app-skb/front/knowledge-space"
        
        headers = {
            'Content-Type': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'accept-encoding': 'gzip, deflate'
        }
        
        try:
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                # Check redirect history
                if response.history:
                    for i, hist_resp in enumerate(response.history):
                        logger.info(f"  {i+1}. {hist_resp.status} -> {hist_resp.url}")
                        set_cookie = hist_resp.headers.get('Set-Cookie', '')
                
                if response.status == 200:
                    # Check and update session
                    session_updated = False
                    
                    # Check final response cookie
                    set_cookie_header = response.headers.get('Set-Cookie', '')
                    if set_cookie_header:
                        # Use JMeter script regex logic
                        cookie_match = re.search(r'([^;]+); Path=', set_cookie_header)
                        if cookie_match:
                            self.cookie_session = cookie_match.group(1)
                            session_updated = True
                        else:
                            # Fallback: extract first cookie
                            cookie_match = re.search(r'([^;]+)', set_cookie_header)
                            if cookie_match:
                                self.cookie_session = cookie_match.group(1)
                                session_updated = True
                    
                    # Check cookies in redirect history
                    if not session_updated and response.history:
                        for hist_resp in response.history:
                            set_cookie = hist_resp.headers.get('Set-Cookie', '')
                            if set_cookie:
                                # Use JMeter script regex logic
                                cookie_match = re.search(r'([^;]+); Path=', set_cookie)
                                if cookie_match:
                                    self.cookie_session = cookie_match.group(1)
                                    session_updated = True
                                    break
                                else:
                                    # Fallback
                                    cookie_match = re.search(r'([^;]+)', set_cookie)
                                    if cookie_match:
                                        self.cookie_session = cookie_match.group(1)
                                        session_updated = True
                                        break
                    
                    # If no cookie from response headers, check session object
                    if not session_updated:
                        cookies = self.session.cookie_jar
                        if cookies:
                            cookie_parts = []
                            for cookie in cookies:
                                cookie_parts.append(f"{cookie.key}={cookie.value}")
                            
                            if cookie_parts:
                                new_cookie_session = "; ".join(cookie_parts)
                                if new_cookie_session != self.cookie_session:
                                    self.cookie_session = new_cookie_session
                                    session_updated = True
                    
                    if not session_updated:
                        logger.info("Keeping existing session (no new cookie)")
                    
                    return True, None
                else:
                    logger.error(f"Access knowledge homepage failed: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text[:200]}...")
                    return False, ErrorCode.AUTH_ACCESS_HOMEPAGE_FAILED
                    
        except Exception as e:
            logger.error(f"Access knowledge homepage exception: {e}")
            return False, ErrorCode.AUTH_ACCESS_HOMEPAGE_FAILED
    
    async def get_menu_info(self, intranet: Optional[bool] = None) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Get menu information
        
        Args:
            intranet: Whether to use intranet address
            
        Returns:
            Tuple of (success, error_code)
        """
        await self.ensure_session()

        use_intranet = intranet if intranet is not None else config_manager.server.intranet
        
        if use_intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}/ckb/auth/info"
        else:
            base_url = config_manager.server.external_base_url
            url = f"{base_url}/ckb/auth/info"
        
        headers = {
            'Content-Type': 'application/json',
            'Cookie': self.cookie_session,
        }
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        user_data = data.get('data', {})
                        self.user_id = user_data.get('id')
                        
                        if self.user_id:
                            logger.debug(f"User ID retrieved: {self.user_id}")
                            return True, None
                        else:
                            logger.error("No user ID in response")
                            logger.error(f"Response data: {data}")
                            return False, ErrorCode.AUTH_GET_MENU_FAILED
                            
                    except Exception as e:
                        logger.error(f"JSON parsing failed: {e}")
                        response_text = await response.text()
                        logger.error(f"Response body: {response_text[:200]}...")
                        return False, ErrorCode.AUTH_GET_MENU_FAILED
                        
                else:
                    logger.error(f"Get menu info failed: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text[:200]}...")
                    return False, ErrorCode.AUTH_GET_MENU_FAILED
                    
        except Exception as e:
            logger.error(f"Get menu info exception: {e}")
            return False, ErrorCode.AUTH_GET_MENU_FAILED
    
    def is_logged_in(self) -> bool:
        """
        Check if logged in
        
        Returns:
            bool: Whether logged in
        """
        has_basic_info = (self.user_id is not None and 
                         self.tenant_id is not None)
        
        has_session = (self.cookie_session is not None or 
                      (self.session and len(self.session.cookie_jar) > 0))
        
        return has_basic_info and has_session
    
    async def logout(self) -> None:
        """Logout and clean session information"""
        logger.info("Cleaning login state...")
        self.publickey = None
        self.tenant_id = None
        self.redirect_ip = None
        self.redirect_port = None
        self.redirect_path = None
        self.cookie_session = None
        self.user_id = None
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Logout completed")
    
    async def login(self) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Complete login flow
        
        Returns:
            Tuple of (success, error_code)
        """
        # Ensure session is created
        await self.ensure_session()
        
        # Reset login-related state variables
        self.publickey = None
        self.tenant_id = None
        self.redirect_ip = None
        self.redirect_port = None
        self.redirect_path = None
        self.cookie_session = None
        self.user_id = None
        
        # Execute login steps
        login_steps = [
            ("1. Get config", self.get_config),
            ("2. Get public key", self.get_public_key),
            ("3. Get tenant info", self.get_tenant_info),
            ("4. Login UAP", self.login_uap),
            ("5. Redirect login", self.redirect_login),
            ("6. Access knowledge homepage", self.access_knowledge_homepage),
            ("7. Get menu info", self.get_menu_info)
        ]
        
        for step_name, step_func in login_steps:
            try:
                # Use config value (None means use config default)
                success, error_code = await step_func(intranet=None)
                if not success:
                    logger.error(f"Login failed at step: {step_name}, error_code: {error_code.code if error_code else 'Unknown'}")
                    return False, error_code
                logger.info(f"{step_name} successful")
                
                # Short delay to avoid requests too fast
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"{step_name} execution error: {e}")
                return False, ErrorCode.AUTH_UAP_FAILED
        return True, None
    
    async def login_flow(self) -> Tuple[bool, Optional[ErrorCode]]:
        """
        Complete login flow - maintained for backward compatibility
        
        Returns:
            Tuple of (success, error_code)
        """
        return await self.login()


async def login_knowledge() -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[ErrorCode]]:
    """
    Login to knowledge base system
    
    Returns:
        Tuple of (success, user_id, tenant_id, cookie_session, error_code)
    """
    async with LoginManager() as login_manager:
        # Test login
        success, error_code = await login_manager.login()
        
        if success:
            logger.debug(f"Login test successful\n"
            f"- User ID: {login_manager.user_id}\n"
            f"- Tenant ID: {login_manager.tenant_id}\n"
            f"- Redirect server: {login_manager.redirect_ip}:{login_manager.redirect_port}"
            )
            return True, login_manager.user_id, login_manager.tenant_id, login_manager.cookie_session, None
        else:
            logger.error(f"Login test failed: {error_code.code if error_code else 'Unknown'} - {error_code.message if error_code else 'Unknown error'}")
            return False, None, None, None, error_code or ErrorCode.AUTH_UAP_FAILED
