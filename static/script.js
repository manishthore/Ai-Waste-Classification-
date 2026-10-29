document.addEventListener("DOMContentLoaded", function () {

    const fileInput = document.getElementById("wasteImage");
    const form = document.getElementById("predictionForm");
    const button = document.getElementById("classifyButton");

    if (!fileInput || !form || !button) {
        return;
    }


    fileInput.addEventListener("change", function () {

        if (this.files && this.files.length > 0) {

            button.disabled = false;

        } else {

            button.disabled = true;

        }

    });


    form.addEventListener("submit", function () {

        if (!fileInput.files || fileInput.files.length === 0) {

            alert("Please select a waste image.");

            return false;

        }

        button.disabled = true;

        button.innerText = "⏳ Classifying...";

    });

});