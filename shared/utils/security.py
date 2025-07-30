"""
보안 관련 유틸리티
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from cryptography.fernet import Fernet
import os


class JWTManager:
    """JWT 토큰 관리"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.algorithm = "HS256"
        self.access_token_expire_hours = 24
    
    def _generate_secret_key(self) -> str:
        """시크릿 키 생성"""
        return secrets.token_urlsafe(32)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """액세스 토큰 생성"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.access_token_expire_hours)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("토큰이 만료되었습니다")
        except jwt.JWTError:
            raise Exception("유효하지 않은 토큰입니다")


class PasswordManager:
    """비밀번호 관리"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """비밀번호 해싱"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        try:
            salt, stored_hash = hashed_password.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash.hex() == stored_hash
        except ValueError:
            return False


class DataEncryption:
    """데이터 암호화"""
    
    def __init__(self, key: bytes = None):
        if key is None:
            key = os.getenv("ENCRYPTION_KEY")
            if key:
                key = key.encode()
            else:
                key = Fernet.generate_key()
        
        self.cipher_suite = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """데이터 암호화"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
        return decrypted_data.decode()


def generate_api_key() -> str:
    """API 키 생성"""
    return secrets.token_urlsafe(32)


def validate_api_key(api_key: str, valid_keys: list) -> bool:
    """API 키 검증"""
    return api_key in valid_keys