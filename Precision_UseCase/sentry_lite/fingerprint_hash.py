# sentry_lite/fingerprint_hash.py 

import hashlib 

from datetime import datetime 

 

def generate_sponsor_id(fingerprint_bytes: bytes) -> str: 

    salt = str(datetime.utcnow()).encode("utf-8") 

    return hashlib.sha256(fingerprint_bytes + salt).hexdigest() 