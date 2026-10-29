# ============================================================
# AI-BASED SMART WASTE CLASSIFICATION SYSTEM
# Complete Flask Application
# Environmental Impact Added
# ============================================================

import os
import uuid
from functools import wraps
from datetime import datetime

import numpy as np
import mysql.connector

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "ai_waste_classification_secret_key_2026"

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MODEL_PATH = "waste_model.keras"

IMAGE_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# WASTE CLASSES
# MUST MATCH MODEL TRAINING ORDER
# ============================================================

CLASS_NAMES = [
    "cardboard",
    "e_waste",
    "glass",
    "metal",
    "other",
    "paper",
    "plastic",
    "trash"
]


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    # --------------------------------------------------------
    # CARDBOARD
    # --------------------------------------------------------

    "cardboard": {

        "name": "Cardboard",

        "icon": "📦",

        "recommendation":
            "Keep cardboard clean and dry. Flatten boxes before recycling.",

        "recycling_uses":
            "Packaging boxes, paperboard, storage boxes and recycled paper products.",

        "recycling_process":
            "Collected cardboard is sorted, shredded, pulped, cleaned and converted into new paper or cardboard products.",

        "solutions":
            "Reuse boxes, flatten them and send them to a paper/cardboard recycling collection center.",

        "action":
            "Flatten and recycle.",

        "bin":
            "Blue / Dry Waste Bin",

        "environmental_impact":
            "Recycling cardboard reduces the demand for fresh raw materials, saves energy and water, and decreases the amount of waste sent to landfills."
    },


    # --------------------------------------------------------
    # E-WASTE
    # --------------------------------------------------------

    "e_waste": {

        "name": "E-Waste",

        "icon": "💻",

        "recommendation":
            "Do not throw electronic waste into normal household bins.",

        "recycling_uses":
            "Recovery of metals, plastics, glass and reusable electronic components.",

        "recycling_process":
            "Electronic devices are collected, dismantled, separated and processed to recover valuable materials.",

        "solutions":
            "Give electronic waste to an authorized e-waste collection or recycling center.",

        "action":
            "Send to authorized e-waste recycler.",

        "bin":
            "E-Waste Collection",

        "environmental_impact":
            "Improper e-waste disposal can release harmful substances into soil and water. Proper recycling helps recover valuable materials and reduces environmental pollution."
    },


    # --------------------------------------------------------
    # GLASS
    # --------------------------------------------------------

    "glass": {

        "name": "Glass",

        "icon": "🍾",

        "recommendation":
            "Separate glass bottles and containers from other waste.",

        "recycling_uses":
            "New bottles, jars, glass containers and construction materials.",

        "recycling_process":
            "Glass is sorted by type and color, cleaned, crushed into cullet and melted to produce new glass products.",

        "solutions":
            "Reuse bottles when possible and send broken or unwanted glass to a recycling center.",

        "action":
            "Separate and recycle.",

        "bin":
            "Blue / Glass Collection",

        "environmental_impact":
            "Recycling glass reduces the need for new raw materials, saves energy and decreases the amount of glass waste sent to landfills."
    },


    # --------------------------------------------------------
    # METAL
    # --------------------------------------------------------

    "metal": {

        "name": "Metal",

        "icon": "🥫",

        "recommendation":
            "Separate metal cans and objects from general waste.",

        "recycling_uses":
            "New cans, automobile parts, construction materials and metal products.",

        "recycling_process":
            "Metal is collected, sorted, compressed, melted and manufactured into new metal products.",

        "solutions":
            "Clean metal containers and send them to a dry-waste or metal recycling collection center.",

        "action":
            "Clean and recycle.",

        "bin":
            "Blue / Dry Waste Bin",

        "environmental_impact":
            "Recycling metal reduces mining activities, saves energy and conserves natural resources while reducing landfill waste."
    },


    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    "other": {

        "name": "Other / Non-Waste",

        "icon": "♻️",

        "recommendation":
            "The image could not be confidently identified as one of the specific waste categories.",

        "recycling_uses":
            "Depends on the actual material. Further manual identification may be required.",

        "recycling_process":
            "The correct process depends on the material identified during manual inspection.",

        "solutions":
            "Check the material manually and dispose of it according to local waste-management guidelines.",

        "action":
            "Inspect and dispose responsibly.",

        "bin":
            "General / Appropriate Collection",

        "environmental_impact":
            "The environmental impact depends on the material. Proper identification and responsible disposal can help prevent pollution and reduce unnecessary landfill waste."
    },


    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    "paper": {

        "name": "Paper",

        "icon": "📄",

        "recommendation":
            "Keep paper clean and dry and separate it from wet waste.",

        "recycling_uses":
            "New paper, notebooks, newspapers, packaging and tissue products.",

        "recycling_process":
            "Paper is sorted, shredded, mixed with water to create pulp, cleaned and processed into new paper products.",

        "solutions":
            "Reuse paper when possible and send clean paper to a recycling collection center.",

        "action":
            "Keep dry and recycle.",

        "bin":
            "Blue / Dry Waste Bin",

        "environmental_impact":
            "Paper recycling reduces the need for fresh wood, saves water and energy, and helps reduce landfill waste."
    },


    # --------------------------------------------------------
    # PLASTIC
    # --------------------------------------------------------

    "plastic": {

        "name": "Plastic",

        "icon": "🧴",

        "recommendation":
            "Separate plastic containers and packaging from wet waste.",

        "recycling_uses":
            "Plastic containers, furniture, packaging materials, fibers and other plastic products.",

        "recycling_process":
            "Plastic is sorted by type, cleaned, shredded, melted and converted into reusable plastic material.",

        "solutions":
            "Reduce single-use plastic, reuse containers and send recyclable plastic to a proper collection center.",

        "action":
            "Clean, separate and recycle.",

        "bin":
            "Blue / Dry Waste Bin",

        "environmental_impact":
            "Proper plastic recycling reduces plastic pollution, protects ecosystems and decreases the amount of plastic entering landfills and natural environments."
    },


    # --------------------------------------------------------
    # TRASH
    # --------------------------------------------------------

    "trash": {

        "name": "Trash",

        "icon": "🗑️",

        "recommendation":
            "General trash should be separated from recyclable materials whenever possible.",

        "recycling_uses":
            "Depends on the material. Some components may still be recoverable after sorting.",

        "recycling_process":
            "Trash is sorted to recover recyclable materials. Remaining waste is handled according to local waste-management practices.",

        "solutions":
            "Separate recyclable materials before disposing of general waste.",

        "action":
            "Separate recyclables and dispose of remaining waste properly.",

        "bin":
            "General Waste Bin",

        "environmental_impact":
            "Poorly managed general waste can contribute to land and water pollution. Separating recyclable materials helps reduce its environmental impact."
    }
}


# ============================================================
# ADMIN DETAILS
# ============================================================

ADMIN_EMAIL = "admin@wasteai.com"
ADMIN_PASSWORD = "admin123"


# ============================================================
# AI MODEL
# ============================================================

model = None


def load_ai_model():

    global model

    try:

        print()
        print("=" * 60)
        print("LOADING AI MODEL")
        print("=" * 60)

        if not os.path.exists(MODEL_PATH):

            print("❌ Model file not found:")
            print(MODEL_PATH)

            return False

        model = load_model(
            MODEL_PATH,
            compile=False
        )

        print("✅ AI model loaded successfully.")

        print(
            "Model input shape:",
            model.input_shape
        )

        print(
            "Model output shape:",
            model.output_shape
        )

        expected_classes = len(CLASS_NAMES)

        model_classes = model.output_shape[-1]

        print(
            "Expected classes:",
            expected_classes
        )

        print(
            "Model classes:",
            model_classes
        )

        if model_classes != expected_classes:

            print("❌ MODEL CLASS COUNT DOES NOT MATCH.")

            model = None

            return False

        print("✅ Model class count verified.")

        return True

    except Exception as e:

        print()
        print("❌ MODEL LOADING ERROR")
        print(str(e))

        model = None

        return False


load_ai_model()


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_waste_classification"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    try:

        connection = mysql.connector.connect(

            host=DB_HOST,

            port=DB_PORT,

            user=DB_USER,

            password=DB_PASSWORD,

            database=DB_NAME

        )

        return connection

    except mysql.connector.Error as e:

        print(
            "❌ MySQL connection error:",
            e
        )

        return None


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    try:

        connection = mysql.connector.connect(

            host=DB_HOST,

            port=DB_PORT,

            user=DB_USER,

            password=DB_PASSWORD

        )

        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`"
        )

        cursor.close()

        connection.close()

        print("✅ Database checked.")

    except mysql.connector.Error as e:

        print(
            "❌ Database creation error:",
            e
        )


# ============================================================
# CREATE TABLES
# ============================================================

def initialize_database():

    create_database()

    connection = get_db_connection()

    if connection is None:

        print(
            "❌ Database initialization failed."
        )

        return

    cursor = connection.cursor()

    try:

        # ----------------------------------------------------
        # USERS TABLE
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (

                id INT AUTO_INCREMENT PRIMARY KEY,

                name VARCHAR(100) NOT NULL,

                email VARCHAR(255) UNIQUE NOT NULL,

                password VARCHAR(255) NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
        """)


        # ----------------------------------------------------
        # PREDICTIONS TABLE
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT NOT NULL,

                image_file VARCHAR(255) NOT NULL,

                prediction VARCHAR(100) NOT NULL,

                confidence FLOAT NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE

            )
        """)

        connection.commit()

        print(
            "✅ Database initialized."
        )

    except mysql.connector.Error as e:

        print(
            "❌ Table creation error:",
            e
        )

    finally:

        cursor.close()

        connection.close()


initialize_database()


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login first.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return function(*args, **kwargs)

    return decorated_function


# ============================================================
# ADMIN REQUIRED
# ============================================================

def admin_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if not session.get("admin_logged_in"):

            return redirect(
                url_for("admin_login")
            )

        return function(*args, **kwargs)

    return decorated_function


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    if not filename:

        return False

    if "." not in filename:

        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        if not name or not email or not password:

            flash(
                "Please fill all required fields.",
                "danger"
            )

            return render_template(
                "register.html"
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return render_template(
                "register.html"
            )


        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )

            return render_template(
                "register.html"
            )


        connection = get_db_connection()

        if connection is None:

            flash(
                "Database connection failed.",
                "danger"
            )

            return render_template(
                "register.html"
            )


        cursor = connection.cursor()

        try:

            cursor.execute(
                "SELECT id FROM users WHERE email = %s",
                (email,)
            )

            existing_user = cursor.fetchone()


            if existing_user:

                flash(
                    "Email is already registered.",
                    "danger"
                )

                return render_template(
                    "register.html"
                )


            hashed_password = generate_password_hash(
                password
            )


            cursor.execute(
                """
                INSERT INTO users
                (name, email, password)
                VALUES (%s, %s, %s)
                """,

                (
                    name,
                    email,
                    hashed_password
                )
            )

            connection.commit()


            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )


        except mysql.connector.Error as e:

            connection.rollback()

            print(
                "Registration error:",
                e
            )

            flash(
                "Registration failed.",
                "danger"
            )

        finally:

            cursor.close()

            connection.close()


    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if not email or not password:

            flash(
                "Please enter email and password.",
                "danger"
            )

            return render_template(
                "login.html"
            )


        connection = get_db_connection()

        if connection is None:

            flash(
                "Database connection failed.",
                "danger"
            )

            return render_template(
                "login.html"
            )


        cursor = connection.cursor(
            dictionary=True
        )


        try:

            cursor.execute(
                """
                SELECT id, name, email, password
                FROM users
                WHERE email = %s
                """,

                (email,)
            )

            user = cursor.fetchone()


            if user and check_password_hash(
                user["password"],
                password
            ):

                session.clear()

                session["user_id"] = user["id"]

                session["user_name"] = user["name"]

                session["user_email"] = user["email"]


                return redirect(
                    url_for("dashboard")
                )


            flash(
                "Invalid email or password.",
                "danger"
            )


        except mysql.connector.Error as e:

            print(
                "Login error:",
                e
            )

            flash(
                "Login failed.",
                "danger"
            )

        finally:

            cursor.close()

            connection.close()


    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    connection = get_db_connection()

    if connection is None:

        return """
        <h2>Dashboard Error</h2>
        <p>Database connection failed.</p>
        """

    cursor = connection.cursor(
        dictionary=True
    )


    try:

        user_id = session["user_id"]


        # ----------------------------------------------------
        # TOTAL PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            """,

            (user_id,)
        )

        result = cursor.fetchone()

        total_predictions = (

            result["total"]

            if result

            else 0

        )


        # ----------------------------------------------------
        # CATEGORY COUNTS
        # ----------------------------------------------------

        category_counts = {}

        for category in CLASS_NAMES:

            category_counts[category] = 0


        cursor.execute(
            """
            SELECT prediction, COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            GROUP BY prediction
            """,

            (user_id,)
        )


        rows = cursor.fetchall()


        for row in rows:

            category = row["prediction"]

            if category in category_counts:

                category_counts[
                    category
                ] = row["total"]


        # ----------------------------------------------------
        # CATEGORY PERCENTAGES
        # ----------------------------------------------------

        category_percentages = {}


        for category in CLASS_NAMES:

            if total_predictions > 0:

                category_percentages[category] = round(

                    (
                        category_counts[category]
                        /
                        total_predictions
                    ) * 100,

                    1

                )

            else:

                category_percentages[category] = 0


        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                image_file,
                prediction,
                confidence,
                created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT 5
            """,

            (user_id,)
        )


        recent_predictions = cursor.fetchall()


        # ----------------------------------------------------
        # AVERAGE CONFIDENCE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT AVG(confidence) AS average_confidence
            FROM predictions
            WHERE user_id = %s
            """,

            (user_id,)
        )


        avg_result = cursor.fetchone()


        average_confidence = (

            round(
                avg_result["average_confidence"],
                2
            )

            if (
                avg_result
                and
                avg_result["average_confidence"]
                is not None
            )

            else 0

        )


        return render_template(

            "dashboard.html",

            total_predictions=total_predictions,

            category_counts=category_counts,

            category_percentages=category_percentages,

            recent_predictions=recent_predictions,

            average_confidence=average_confidence

        )


    except mysql.connector.Error as e:

        print(
            "Dashboard database error:",
            e
        )

        return """
        <h2>Dashboard Error</h2>
        <p>Database error occurred.</p>
        <p>Check the VS Code terminal.</p>
        """


    except Exception as e:

        print(
            "Dashboard error:",
            e
        )

        return f"""
        <h2>Dashboard Error</h2>
        <p>{str(e)}</p>
        <p>Check the VS Code terminal.</p>
        """


    finally:

        cursor.close()

        connection.close()


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    print()
    print("=" * 60)
    print("PREDICTION REQUEST RECEIVED")
    print("=" * 60)


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        print(
            "❌ AI model is not loaded."
        )

        return """
        <h2>AI Model Error</h2>
        <p>The AI model could not be loaded.</p>
        <p>Check waste_model.keras.</p>
        """, 500


    # --------------------------------------------------------
    # GET FILE
    # --------------------------------------------------------

    file = request.files.get("file")


    if file is None:

        print(
            "❌ No file received."
        )

        return """
        <h2>Upload Error</h2>
        <p>No image was received.</p>
        <p>Please select an image and try again.</p>
        """, 400


    if file.filename == "":

        print(
            "❌ Empty filename."
        )

        return """
        <h2>Upload Error</h2>
        <p>No image selected.</p>
        """, 400


    print(
        "File received:",
        file.filename
    )


    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        print(
            "❌ Invalid file type."
        )

        return """
        <h2>Invalid File</h2>
        <p>Please upload JPG, JPEG, PNG or WEBP.</p>
        """, 400


    # --------------------------------------------------------
    # UNIQUE FILE NAME
    # --------------------------------------------------------

    original_name = secure_filename(
        file.filename
    )

    extension = original_name.rsplit(
        ".",
        1
    )[1].lower()


    unique_filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        file.save(filepath)

        print(
            "Image saved:",
            filepath
        )

    except Exception as e:

        print(
            "❌ Image save error:",
            e
        )

        return """
        <h2>Upload Error</h2>
        <p>Could not save the uploaded image.</p>
        """, 500


    # --------------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------------

    try:

        image = Image.open(
            filepath
        ).convert("RGB")


        image = image.resize(
            IMAGE_SIZE
        )


        image_array = np.array(
            image,
            dtype=np.float32
        )


        image_array = preprocess_input(
            image_array
        )


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Image prepared:",
            image_array.shape
        )


    except Exception as e:

        print(
            "❌ Image processing error:",
            e
        )

        return """
        <h2>Image Processing Error</h2>
        <p>The selected image could not be processed.</p>
        """, 500


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            image_array,
            verbose=0
        )


        probabilities = prediction[0]


        predicted_index = int(
            np.argmax(probabilities)
        )


        confidence = float(
            probabilities[predicted_index]
            * 100
        )


        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        top_predictions = []


        for index in top_indices:

            top_predictions.append({

                "class": CLASS_NAMES[index],

                "name": WASTE_INFO[
                    CLASS_NAMES[index]
                ]["name"],

                "confidence": round(

                    float(
                        probabilities[index]
                        * 100
                    ),

                    2

                )

            })


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        print()
        print("AI RESULT")
        print("-" * 40)

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )


        # ----------------------------------------------------
        # CONFIDENCE STATUS
        # ----------------------------------------------------

        if confidence >= 85:

            confidence_status = "High Confidence"

            status_message = (
                "The AI model is highly confident "
                "about this classification."
            )


        elif confidence >= 70:

            confidence_status = "Medium Confidence"

            status_message = (
                "The AI model has reasonable confidence "
                "about this classification."
            )


        else:

            confidence_status = "Low Confidence"

            status_message = (
                "The AI model has low confidence. "
                "The result should be manually verified."
            )


        # ----------------------------------------------------
        # GET WASTE INFORMATION
        # ----------------------------------------------------

        info = WASTE_INFO.get(

            predicted_class,

            WASTE_INFO["other"]

        )


        # ----------------------------------------------------
        # SAVE PREDICTION TO MYSQL
        # ----------------------------------------------------

        connection = get_db_connection()


        if connection is None:

            print(
                "❌ Could not connect to database."
            )

            return """
            <h2>Database Error</h2>
            <p>Could not save prediction.</p>
            """, 500


        cursor = connection.cursor()


        try:

            cursor.execute(
                """
                INSERT INTO predictions
                (
                    user_id,
                    image_file,
                    prediction,
                    confidence
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,

                (
                    session["user_id"],

                    unique_filename,

                    predicted_class,

                    round(
                        confidence,
                        2
                    )

                )

            )


            connection.commit()


            print(
                "✅ Prediction saved to database."
            )


        except mysql.connector.Error as e:

            connection.rollback()

            print(
                "❌ Prediction database error:",
                e
            )

            return """
            <h2>Database Error</h2>
            <p>Prediction was generated but could not be saved.</p>
            """, 500


        finally:

            cursor.close()

            connection.close()


        # ----------------------------------------------------
        # SHOW RESULT
        # ----------------------------------------------------

        return render_template(

            "result.html",

            prediction=info["name"],

            confidence=round(
                confidence,
                2
            ),

            image_file=unique_filename,

            recommendation=info["recommendation"],

            recycling_uses=info["recycling_uses"],

            recycling_process=info["recycling_process"],

            solutions=info["solutions"],

            action=info["action"],

            recommended_bin=info["bin"],

            environmental_impact=info[
                "environmental_impact"
            ],

            waste_icon=info["icon"],

            confidence_status=confidence_status,

            status_message=status_message,

            top_predictions=top_predictions,

            created_at=datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )

        )


    except Exception as e:

        print()
        print(
            "❌ PREDICTION ERROR"
        )
        print("=" * 60)
        print(
            str(e)
        )
        print("=" * 60)


        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <p>Please check the VS Code terminal.</p>
        """, 500


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    connection = get_db_connection()


    if connection is None:

        return """
        <h2>History Error</h2>
        <p>Database connection failed.</p>
        """


    cursor = connection.cursor(
        dictionary=True
    )


    try:

        cursor.execute(
            """
            SELECT
                id,
                image_file,
                prediction,
                confidence,
                created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY id DESC
            """,

            (
                session["user_id"],
            )

        )


        predictions = cursor.fetchall()


        return render_template(

            "history.html",

            predictions=predictions

        )


    except mysql.connector.Error as e:

        print(
            "History error:",
            e
        )

        return """
        <h2>History Error</h2>
        <p>Could not load prediction history.</p>
        """


    finally:

        cursor.close()

        connection.close()


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if (
            email == ADMIN_EMAIL
            and
            password == ADMIN_PASSWORD
        ):

            session["admin_logged_in"] = True

            session["admin_email"] = email


            return redirect(
                url_for("admin_dashboard")
            )


        flash(
            "Invalid admin credentials.",
            "danger"
        )


    return render_template(
        "admin_login.html"
    )


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.pop(
        "admin_logged_in",
        None
    )

    session.pop(
        "admin_email",
        None
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    connection = get_db_connection()


    if connection is None:

        return """
        <h2>Admin Dashboard Error</h2>
        <p>Database connection failed.</p>
        """


    cursor = connection.cursor(
        dictionary=True
    )


    try:

        # ----------------------------------------------------
        # TOTAL USERS
        # ----------------------------------------------------

        cursor.execute(
            "SELECT COUNT(*) AS total FROM users"
        )

        total_users = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # TOTAL PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            "SELECT COUNT(*) AS total FROM predictions"
        )

        total_predictions = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # AVERAGE CONFIDENCE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT AVG(confidence) AS average
            FROM predictions
            """
        )

        average_result = cursor.fetchone()


        average_confidence = (

            round(
                average_result["average"],
                2
            )

            if (
                average_result
                and
                average_result["average"]
                is not None
            )

            else 0

        )


        # ----------------------------------------------------
        # CATEGORY COUNTS
        # ----------------------------------------------------

        category_counts = {}

        for category in CLASS_NAMES:

            category_counts[category] = 0


        cursor.execute(
            """
            SELECT
                prediction,
                COUNT(*) AS total
            FROM predictions
            GROUP BY prediction
            """
        )


        rows = cursor.fetchall()


        for row in rows:

            category = row["prediction"]


            if category in category_counts:

                category_counts[
                    category
                ] = row["total"]


        # ----------------------------------------------------
        # CATEGORY PERCENTAGES
        # ----------------------------------------------------

        category_percentages = {}


        for category in CLASS_NAMES:

            if total_predictions > 0:

                category_percentages[
                    category
                ] = round(

                    (
                        category_counts[
                            category
                        ]
                        /
                        total_predictions
                    ) * 100,

                    1

                )

            else:

                category_percentages[
                    category
                ] = 0


        # ----------------------------------------------------
        # USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                created_at
            FROM users
            ORDER BY id DESC
            """
        )


        users = cursor.fetchall()


        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                p.id,
                p.image_file,
                p.prediction,
                p.confidence,
                p.created_at,
                u.name,
                u.email
            FROM predictions p
            LEFT JOIN users u
            ON p.user_id = u.id
            ORDER BY p.id DESC
            LIMIT 20
            """
        )


        recent_predictions = cursor.fetchall()


        return render_template(

            "admin.html",

            total_users=total_users,

            total_predictions=total_predictions,

            average_confidence=average_confidence,

            category_counts=category_counts,

            category_percentages=category_percentages,

            users=users,

            recent_predictions=recent_predictions

        )


    except mysql.connector.Error as e:

        print(
            "Admin dashboard error:",
            e
        )

        return f"""
        <h2>Admin Dashboard Error</h2>
        <p>{str(e)}</p>
        """


    finally:

        cursor.close()

        connection.close()


# ============================================================
# DELETE USER
# ============================================================

@app.route(
    "/admin/delete-user/<int:user_id>",
    methods=["POST"]
)
@admin_required
def delete_user(user_id):

    connection = get_db_connection()


    if connection is None:

        return redirect(
            url_for("admin_dashboard")
        )


    cursor = connection.cursor()


    try:

        cursor.execute(
            "DELETE FROM users WHERE id = %s",
            (user_id,)
        )

        connection.commit()


        flash(
            "User deleted successfully.",
            "success"
        )


    except mysql.connector.Error as e:

        connection.rollback()

        print(
            "Delete user error:",
            e
        )

        flash(
            "Could not delete user.",
            "danger"
        )


    finally:

        cursor.close()

        connection.close()


    return redirect(
        url_for("admin_dashboard")
    )


# ============================================================
# DELETE PREDICTION
# ============================================================

@app.route(
    "/admin/delete-prediction/<int:prediction_id>",
    methods=["POST"]
)
@admin_required
def delete_prediction(prediction_id):

    connection = get_db_connection()


    if connection is None:

        return redirect(
            url_for("admin_dashboard")
        )


    cursor = connection.cursor()


    try:

        cursor.execute(
            """
            DELETE FROM predictions
            WHERE id = %s
            """,

            (prediction_id,)
        )


        connection.commit()


        flash(
            "Prediction deleted successfully.",
            "success"
        )


    except mysql.connector.Error as e:

        connection.rollback()

        print(
            "Delete prediction error:",
            e
        )

        flash(
            "Could not delete prediction.",
            "danger"
        )


    finally:

        cursor.close()

        connection.close()


    return redirect(
        url_for("admin_dashboard")
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return """
    <h2>File Too Large</h2>
    <p>Please upload an image smaller than 10 MB.</p>
    """, 413


@app.errorhandler(404)
def page_not_found(error):

    return """
    <h2>404 - Page Not Found</h2>
    <p>The requested page does not exist.</p>
    """, 404


@app.errorhandler(500)
def internal_error(error):

    return """
    <h2>500 - Internal Server Error</h2>
    <p>Please check the VS Code terminal.</p>
    """, 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI WASTE CLASSIFICATION SYSTEM")
    print("=" * 60)

    print()

    if model is not None:

        print("✅ AI model ready.")

    else:

        print("❌ AI model NOT ready.")

    print()

    print("=" * 60)
    print("SERVER STARTING")
    print("=" * 60)

    print()

    print(
        "Website:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "Admin Login:"
    )

    print(
        "Email:",
        ADMIN_EMAIL
    )

    print(
        "Password:",
        ADMIN_PASSWORD
    )

    print()

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )