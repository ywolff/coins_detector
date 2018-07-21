from src.constants.coins import COINS


def RGB_to_BGR(RGB_values):
    return (RGB_values[2], RGB_values[1], RGB_values[0])


def format_CSS_RGB_color(RGB_values):
    return 'rgb(' + ','.join((str(value) for value in RGB_values)) + ')'


def format_detected_coins_summary(detected_coins):
    detected_coins_counts = {
        value: 0 for value in COINS
    }
    for coin in detected_coins:
        detected_coins_counts[coin['value']] += 1
    return [
        (detected_coins_counts[value], value)
        for value in detected_coins_counts
        if detected_coins_counts[value] > 0
    ]


def format_euro_value(value):
    number_of_euros = int(value)
    number_of_cents = round(100 * (value - number_of_euros))

    if number_of_euros == 0:
        return f"{number_of_cents} cent{'s' if number_of_cents > 1 else ''}"
    if number_of_cents == 0:
        return f'{number_of_euros} €'

    return f"{number_of_euros} € {'%02d' % number_of_cents}"
