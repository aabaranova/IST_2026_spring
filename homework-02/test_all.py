import doctest
import caesar
import vigenere
import rsa

print("=" * 50)
print("ТЕСТИРОВАНИЕ")
print("=" * 50)

print("\n1. Caesar cipher:")
doctest.testmod(caesar, verbose=True)

print("\n2. Vigenere cipher:")
doctest.testmod(vigenere, verbose=True)

print("\n3. RSA:")
doctest.testmod(rsa, verbose=True)

print("\n" + "=" * 50)
print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
print("=" * 50)