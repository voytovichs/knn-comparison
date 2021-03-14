def format_time(ns: int) -> str:
    micro = 1000
    milli = micro * 1000
    second = milli * 1000
    minute = second * 60

    if micro >= ns:
        return '{}ns'.format(ns)

    if milli >= ns > micro:
        return '{}μs'.format(ns / micro)

    if second >= ns > milli:
        return '{}ms'.format(ns / milli)

    if minute >= ns > second:
        return '{}sec'.format(int(ns / second))

    return '{}min {}sec'.format(int(ns / minute), int((ns % minute) / second))


def format_bytes(b: int) -> str:
    kb = 1024
    mb = kb * 1024
    gb = mb * 1024

    if kb >= b:
        return '{}b'.format(b)

    if mb >= b > kb:
        return '{}kb'.format(int(b / kb))

    if gb >= b > mb:
        return '{}mb'.format(int(b / mb))

    return '{}g {}mb'.format(int(b / gb), int((b % gb) / mb))
