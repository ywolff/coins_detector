import os
from datetime import datetime

from flask import (
    Flask,
    redirect,
    request,
    render_template,
    send_from_directory
)
from werkzeug.utils import secure_filename

from .constants.coins import COINS_LABELS, COINS_LEGEND_COLORS
from .utils.formatters import format_CSS_RGB_color, format_detected_coins_summary, format_euro_value
from .utils.coins_utils import get_total_value

from .coins_detectors import coins_detectors

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',
                           coins_detectors=coins_detectors,
                           COINS_LABELS=COINS_LABELS)


@app.route('/results', methods=['POST'])
def results():
    image = request.files['image']
    detector_key = request.form.get('detector')
    biggest_radius_coin_value = float(request.form.get('biggest_radius_coin_value'))

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join('uploads', filename)
        output_image_path = os.path.join('uploads', f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}")
        image.save(image_path)

        CoinsDetector = coins_detectors[detector_key]
        detector = CoinsDetector()
        detected_coins = detector.detect_and_visualize(image_path, output_image_path, biggest_radius_coin_value)
        total_value = get_total_value(detected_coins)

        return render_template(
            'results.html',
            detector_name=detector.name,
            output_image_path=output_image_path,
            total_value=format_euro_value(total_value),
            detected_coins_summary=format_detected_coins_summary(detected_coins),
            COINS_LEGEND_COLORS=COINS_LEGEND_COLORS,
            COINS_LABELS=COINS_LABELS,
            format_CSS_RGB_color=format_CSS_RGB_color,
        )
    return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("../uploads",
                               filename)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
