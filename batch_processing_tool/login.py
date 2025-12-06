#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import aiohttp
import asyncio
import json
import base64
import hashlib
import time
import re
import ssl
from urllib.parse import urlparse
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.backends import default_backend
from config import config_manager, logger

class LoginManager:
    def __init__(self):
        # 加载配置文件
        self.uap_ip = config_manager.uap_ip
        self.uap_port = config_manager.uap_port
        self.ckb_ip = config_manager.ckb_ip
        self.ckb_port = config_manager.ckb_port
        self.login_name = config_manager.login_name
        self.password = config_manager.password
        # 运行时变量
        self.session = None
        self.publickey = None
        self.tenant_id = None
        self.redirect_ip = None
        self.redirect_port = None
        self.redirect_path = None
        self.cookie_session = None
        self.user_id = None
    
    def _create_ssl_context(self):
        """创建支持传统SSL重新协商的SSL上下文"""
        ssl_context = ssl.create_default_context()
        # 允许不安全的传统重新协商（仅用于兼容旧服务器）
        ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT
        return ssl_context
    
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        ssl_context = self._create_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def ensure_session(self):
        """确保session已创建"""
        if self.session is None:
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        
    def rsa_encrypt(self, data, public_key_str):
        """RSA加密"""
        try:
            # 清理公钥字符串
            public_key_str = re.sub(r'\s+', '', public_key_str)
            
            # Base64解码
            key_bytes = base64.b64decode(public_key_str)
            
            # 加载公钥
            public_key = serialization.load_der_public_key(key_bytes, backend=default_backend())
            
            # 加密数据
            encrypted = public_key.encrypt(
                data.encode('utf-8'),
                padding.PKCS1v15()
            )
            
            # Base64编码返回
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"RSA加密失败: {e}")
            return None
    
    def md5_hash(self, text):
        """MD5加密"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def get_config(self, intranet: bool = True):
        """获取配置"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/getConfig"
        else:
            url = f"https://ssc.mohrss.gov.cn/corpus_uap_server/getConfig"
        
        try:
            async with self.session.post(url) as response:
                data = await response.json()
                logger.info(f"获取配置 {data}")
                return response.status == 200
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return False
    
    async def get_public_key(self, intranet: bool = True):
        """获取公钥"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/api/v2/getPublicKey"
        else:
            url = f"https://ssc.mohrss.gov.cn/corpus_uap_server/api/v2/getPublicKey"
        
        try:
            async with self.session.post(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.publickey = data.get('data', {}).get('publicKey')
                    logger.info(f"公钥信息 {self.publickey}")
                    return True
                else:
                    logger.error(f"获取公钥失败: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"获取公钥失败: {e}")
            return False
    
    async def get_tenant_info(self, intranet: bool = True):
        """获取租户信息"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.uap_ip}:{self.uap_port}/corpus_uap_server/search/tenants/byName"
        else:
            url = f"https://ssc.mohrss.gov.cn/corpus_uap_server/search/tenants/byName"
        
        # 准备加密数据
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
            logger.error("加密失败")
            return False
        
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
                        return True
                logger.error(f"获取租户信息失败: {data}")
                return False
        except Exception as e:
            logger.error(f"获取租户信息失败: {e}")
            return False
    
    async def login_uap(self, intranet: bool = True):
        """登录UAP"""
        await self.ensure_session()

        if intranet:
            _url = f"http://{self.uap_ip}:{self.uap_port}"
        else:
            _url = f"https://ssc.mohrss.gov.cn"
        url = _url + "/corpus_uap_server/api/v2/login"
    
        if intranet:
            service_url =  f"http://{self.ckb_ip}:{self.ckb_port}/app-skb/auth/info"
            redirect_url = f"http://{self.ckb_ip}:{self.ckb_port}/app-skb/front/knowledge-space"
        else:
            service_url =  f"https://ssc.mohrss.gov.cn/app-skb/auth/info"
            redirect_url = f"https://ssc.mohrss.gov.cn/app-skb/front/knowledge-space"
        
        params = {
            'service': service_url,
            'credentialType': 'Password',
            'redirect': redirect_url,
            'rememberMe': 'true',
            'tenantId': self.tenant_id
        }
        
        # 准备加密数据
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
            return False


        if intranet:
            _url = f"http://{self.uap_ip}:{self.uap_port}"
        else:
            _url = f"https://ssc.mohrss.gov.cn"
        
        headers = {
            'Content-Type': 'application/json',
            'Referer': _url
        }
        
        try:
            # 添加随机延迟
            await asyncio.sleep(1 + (time.time() % 1))
            
            async with self.session.post(url, data=encrypted_data, headers=headers, params=params) as response:
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        redirect_url = data.get('data', {}).get('redirectUrl')
                        
                        if redirect_url:
                            # 解析重定向URL
                            parsed = urlparse(redirect_url)
                            self.redirect_ip = parsed.hostname
                            self.redirect_port = str(parsed.port)
                            self.redirect_path = parsed.path
                            if parsed.query:
                                self.redirect_path += '?' + parsed.query
                            return True
                        else:
                            logger.error("响应中没有重定向URL")
                            logger.error(f"响应数据: {data}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"JSON解析失败: {e}")
                        response_text = await response.text()
                        logger.error(f"响应体: {response_text[:200]}...")
                        return False
                        
                else:
                    logger.error(f"登录UAP失败: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"响应体: {response_text[:200]}...")
                    return False
                    
        except Exception as e:
            logger.error(f"登录UAP异常: {e}")
            return False
    
    async def redirect_login(self, intranet: bool = True):
        """重定向登录"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}{self.redirect_path}"
        else:
            # 使用外网地址，从网络截图可以看到正确的重定向URL
            # 需要先获取CSRF token，然后进行重定向
            url = f"https://ssc.mohrss.gov.cn/corpus_uap_manager/getCsrfToken"
        
        headers = {
            'Content-Type': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'accept-encoding': 'gzip, deflate'
        }
        
        
        try:
            # 允许自动跟随重定向
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                
                # 检查重定向历史
                if response.history:
                    for i, hist_resp in enumerate(response.history):
                        logger.info(f"  {i+1}. {hist_resp.status} -> {hist_resp.url}")
                        # 检查每个重定向响应的cookie
                        set_cookie = hist_resp.headers.get('Set-Cookie', '')
                        if set_cookie:
                            logger.info(f"    Set-Cookie: {set_cookie}")
                
                
                if response.status == 200:
                    # 检查所有响应头部中的cookie
                    session_found = False
                    
                    # 首先检查最终响应的cookie
                    set_cookie_header = response.headers.get('Set-Cookie', '')
                    
                    if set_cookie_header:
                        # 使用JMeter脚本中的正则表达式逻辑
                        cookie_match = re.search(r'([^;]+); Path=', set_cookie_header)
                        if cookie_match:
                            self.cookie_session = cookie_match.group(1)
                            session_found = True
                        else:
                            # 备用方案：提取第一个cookie
                            cookie_match = re.search(r'([^;]+)', set_cookie_header)
                            if cookie_match:
                                self.cookie_session = cookie_match.group(1)
                                session_found = True
                    
                    # 如果最终响应没有cookie，检查重定向历史
                    if not session_found and response.history:
                        for hist_resp in response.history:
                            set_cookie = hist_resp.headers.get('Set-Cookie', '')
                            if set_cookie:
                                # 使用JMeter脚本中的正则表达式逻辑
                                cookie_match = re.search(r'([^;]+); Path=', set_cookie)
                                if cookie_match:
                                    self.cookie_session = cookie_match.group(1)
                                    session_found = True
                                    break
                                else:
                                    # 备用方案
                                    cookie_match = re.search(r'([^;]+)', set_cookie)
                                    if cookie_match:
                                        self.cookie_session = cookie_match.group(1)
                                        session_found = True
                                        break
                    
                    # 如果仍然没有找到cookie，检查session对象中的cookies
                    if not session_found:
                        cookies = self.session.cookie_jar
                        
                        if cookies:
                            # 尝试构造cookie字符串
                            cookie_parts = []
                            for cookie in cookies:
                                cookie_parts.append(f"{cookie.key}={cookie.value}")
                            
                            if cookie_parts:
                                self.cookie_session = "; ".join(cookie_parts)
                                session_found = True
                    
                    if session_found:
                        return True
                    else:
                        logger.error("未能获取到任何session cookie")
                        # 显示响应体的前500字符以供调试
                        response_text = await response.text()
                        logger.error(f"响应体前500字符: {response_text[:500]}")
                        return False
                        
                else:
                    logger.error(f"重定向登录失败: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"响应体: {response_text[:200]}...")
                    return False
                    
        except Exception as e:
            logger.error(f"重定向登录异常: {e}")
            return False
    
    async def access_knowledge_homepage(self, intranet: bool = True):
        """访问知识库主页"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}/app-skb/front/knowledge-space"
        else:
            url = f"https://ssc.mohrss.gov.cn/app-skb/front/knowledge-space"
        
        
        headers = {
            'Content-Type': 'application/json',
            'accept-language': 'zh-CN,zh;q=0.9',
            'accept-encoding': 'gzip, deflate'
        }
        
        
        try:
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                
                # 检查重定向历史
                if response.history:
                    for i, hist_resp in enumerate(response.history):
                        logger.info(f"  {i+1}. {hist_resp.status} -> {hist_resp.url}")
                        set_cookie = hist_resp.headers.get('Set-Cookie', '')
                        
                
                if response.status == 200:
                    # 检查并更新session
                    session_updated = False
                    
                    # 检查最终响应的cookie
                    set_cookie_header = response.headers.get('Set-Cookie', '')
                    if set_cookie_header:
                        # 使用JMeter脚本中的正则表达式逻辑
                        cookie_match = re.search(r'([^;]+); Path=', set_cookie_header)
                        if cookie_match:
                            self.cookie_session = cookie_match.group(1)
                            session_updated = True
                        else:
                            # 备用方案：提取第一个cookie
                            cookie_match = re.search(r'([^;]+)', set_cookie_header)
                            if cookie_match:
                                self.cookie_session = cookie_match.group(1)
                                session_updated = True
                    
                    # 检查重定向历史中的cookie
                    if not session_updated and response.history:
                        for hist_resp in response.history:
                            set_cookie = hist_resp.headers.get('Set-Cookie', '')
                            if set_cookie:
                                # 使用JMeter脚本中的正则表达式逻辑
                                cookie_match = re.search(r'([^;]+); Path=', set_cookie)
                                if cookie_match:
                                    self.cookie_session = cookie_match.group(1)
                                    session_updated = True
                                    break
                                else:
                                    # 备用方案
                                    cookie_match = re.search(r'([^;]+)', set_cookie)
                                    if cookie_match:
                                        self.cookie_session = cookie_match.group(1)
                                        session_updated = True
                                        break
                    
                    # 如果没有从响应头获取到cookie，检查session对象
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
                        logger.info("保持原有Session（无新cookie）")
                    
                    return True
                else:
                    logger.error(f"访问知识库主页失败: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"响应体: {response_text[:200]}...")
                    return False
                    
        except Exception as e:
            logger.error(f"访问知识库主页异常: {e}")
            return False
    
    async def get_menu_info(self, intranet: bool = True):
        """获取菜单信息"""
        await self.ensure_session()

        if intranet:
            url = f"http://{self.redirect_ip}:{self.redirect_port}/ckb/auth/info"
        else:
            url = f"https://ssc.mohrss.gov.cn/ckb/auth/info"
        
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
                            return True
                        else:
                            logger.error("响应中没有用户ID")
                            logger.error(f"响应数据: {data}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"JSON解析失败: {e}")
                        response_text = await response.text()
                        logger.error(f"响应体: {response_text[:200]}...")
                        return False
                        
                else:
                    logger.error(f"获取菜单信息失败: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"响应体: {response_text[:200]}...")
                    return False
                    
        except Exception as e:
            logger.error(f"获取菜单信息异常: {e}")
            return False
    
    def is_logged_in(self):
        """
        检查是否已登录
        返回: bool - 是否已登录
        """
        has_basic_info = (self.user_id is not None and 
                         self.tenant_id is not None)
        
        has_session = (self.cookie_session is not None or 
                      (self.session and len(self.session.cookie_jar) > 0))
        
        
        return has_basic_info and has_session
    
    async def logout(self):
        """
        登出并清理会话信息
        """
        logger.info("清理登录状态...")
        self.publickey = None
        self.tenant_id = None
        self.redirect_ip = None
        self.redirect_port = None
        self.redirect_path = None
        self.cookie_session = None
        self.user_id = None
        
        # 关闭session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("登出完成")
    
    async def login(self):
        """
        独立的登录函数 - 可以单独调用
        返回: bool - 登录是否成功
        """
        
        # 确保session已创建
        await self.ensure_session()
        
        # 重置登录相关的状态变量
        self.publickey = None
        self.tenant_id = None
        self.redirect_ip = None
        self.redirect_port = None
        self.redirect_path = None
        self.cookie_session = None
        self.user_id = None
        
        # 执行登录步骤
        login_steps = [
            ("获取配置", self.get_config),
            ("获取公钥", self.get_public_key),
            ("获取租户信息", self.get_tenant_info),
            ("登录UAP", self.login_uap),
            ("重定向登录", self.redirect_login),
            ("访问知识库主页", self.access_knowledge_homepage),
            ("获取菜单信息", self.get_menu_info)
        ]
        
        for step_name, step_func in login_steps:
            try:
                if not await step_func(intranet=False):
                    logger.error(f"登录失败，停止在: {step_name}")
                    return False
                logger.info(f"{step_name} 成功")
                
        
                
                await asyncio.sleep(0.5)  # 短暂延迟，避免请求过快
            except Exception as e:
                logger.error(f"{step_name} 执行出错: {e}")
                return False
        return True
    
    async def login_flow(self):
        """完整的登录流程 - 保持向后兼容"""
        return await self.login()

async def login_knowledge():
    """仅测试登录功能"""
    async with LoginManager() as login_manager:
        # 测试登录
        if await login_manager.login():
            logger.info("登录测试成功")
            
            # 显示登录信息
            logger.info(f"用户ID: {login_manager.user_id}")
            logger.info(f"租户ID: {login_manager.tenant_id}")
            logger.info(f"重定向服务器: {login_manager.redirect_ip}:{login_manager.redirect_port}")
            
            # 测试登录状态检查
            if login_manager.is_logged_in():
                logger.info("登录状态验证成功")
                return login_manager.user_id,login_manager.tenant_id,login_manager.cookie_session
            else:
                logger.error("登录状态验证失败")
        else:
            logger.error("登录测试失败")

