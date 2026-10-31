def crc_step(crc: int, byte: int) -> int:
    crc ^= byte
    for _ in range(8):
        crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return crc
