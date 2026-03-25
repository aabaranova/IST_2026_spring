def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    keyword = keyword.lower()
    key_index = 0
    
    for char in plaintext:
        if char.isalpha():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            if char.islower():
                start = ord('a')
                ciphertext += chr((ord(char) - start + shift) % 26 + start)
            else:
                start = ord('A')
                ciphertext += chr((ord(char) - start + shift) % 26 + start)
            key_index += 1
        else:
            ciphertext += char
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    keyword = keyword.lower()
    key_index = 0
    
    for char in ciphertext:
        if char.isalpha():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            if char.islower():
                start = ord('a')
                plaintext += chr((ord(char) - start - shift) % 26 + start)
            else:
                start = ord('A')
                plaintext += chr((ord(char) - start - shift) % 26 + start)
            key_index += 1
        else:
            plaintext += char
    return plaintext
