# tempting: a real XOR in bit-mixing code
def mix(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x45D9F3B) & 0xFFFFFFFF
    return h ^ (h >> 16)
