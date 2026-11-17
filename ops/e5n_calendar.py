import calendar, datetime as dt


def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1); first = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=first + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    d = dt.date(year, month, calendar.monthrange(year, month)[1]); return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5: return d - dt.timedelta(days=1)
    if d.weekday() == 6: return d + dt.timedelta(days=1)
    return d


def _easter(year):
    a = year % 19; b, c = divmod(year, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    return dt.date(year, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def short_days(year):
    out = {_nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), _last_weekday(year, 5, 0), _observed(dt.date(year, 7, 4)),
           _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4), _nth_weekday(year, 11, 3, 4) + dt.timedelta(days=1)}
    if year >= 2022: out.add(_observed(dt.date(year, 6, 19)))
    j4 = dt.date(year, 7, 4)
    if j4.weekday() < 5 and j4.weekday() != 0: out.add(j4 - dt.timedelta(days=1))
    x = dt.date(year, 12, 24)
    if x.weekday() < 5 and dt.date(year, 12, 25).weekday() != 5: out.add(x)
    return out


def closed_days(year):
    ny = dt.date(year, 1, 1); x = dt.date(year, 12, 25)
    out = {_easter(year) - dt.timedelta(days=2), x - dt.timedelta(days=1) if x.weekday() == 5 else x + dt.timedelta(days=1) if x.weekday() == 6 else x}
    if ny.weekday() < 5: out.add(ny)
    elif ny.weekday() == 6: out.add(ny + dt.timedelta(days=1))
    return out


def full_session(d):
    return d.weekday() < 5 and d not in closed_days(d.year) and d not in short_days(d.year)


def month_end_flag(d):
    left = 0; x = d + dt.timedelta(days=1)
    while x.month == d.month:
        if full_session(x): left += 1
        x += dt.timedelta(days=1)
    return left < 2
