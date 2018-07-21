def get_total_value(detected_coins):
    return sum([
        coin['value'] for coin in detected_coins
    ])
