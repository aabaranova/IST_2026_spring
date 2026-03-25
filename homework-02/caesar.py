import typing as tp

def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    shift = shift % 26
    for char in plaintext:
        if char.isalpha():
            if char.islower():
                start = ord('a')
                ciphertext += chr((ord(char) - start + shift) % 26 + start)
            else:
                start = ord('A')
                ciphertext += chr((ord(char) - start + shift) % 26 + start)
        else:
            ciphertext += char
    return ciphertext

def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    shift = shift % 26
    for char in ciphertext:
        if char.isalpha():
            if char.islower():
                start = ord('a')
                plaintext += chr((ord(char) - start - shift) % 26 + start)
            else:
                start = ord('A')
                plaintext += chr((ord(char) - start - shift) % 26 + start)
        else:
            plaintext += char
    return plaintext

def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    best_match = 0
    
    for shift in range(26):
        decrypted = decrypt_caesar(ciphertext, shift)
        # Count how many words from dictionary appear in decrypted text
        words = decrypted.split()
        matches = sum(1 for word in words if word.lower() in dictionary)
        if matches > best_match:
            best_match = matches
            best_shift = shift
    
    return best_shift
