document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadForm = document.getElementById('uploadForm');
    const uploadArea = document.getElementById('uploadArea');
    const resultArea = document.getElementById('resultArea');
    const loading = document.getElementById('loading');
    const previewImage = document.getElementById('previewImage');
    const surahName = document.getElementById('surahName');
    const confidence = document.getElementById('confidence');
    const tryAgainBtn = document.getElementById('tryAgainBtn');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Browse button click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFiles);

    // Try again button
    tryAgainBtn.addEventListener('click', resetForm);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('active');
    }

    function unhighlight() {
        dropArea.classList.remove('active');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length) {
            const file = files[0];
            if (file.type.match('image.*')) {
                uploadFile(file);
            } else {
                alert('Please upload an image file (JPEG, PNG)');
            }
        }
    }

    function uploadFile(file) {
        uploadArea.style.display = 'none';
        loading.style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                alert(data.error);
                resetForm();
                return;
            }
            
            previewImage.src = '/static/' + data.image_url;
            surahName.textContent = data.prediction;
            confidence.textContent = data.confidence;
            
            resultArea.style.display = 'flex';
        })
        .catch(error => {
            console.error('Error:', error);
            loading.style.display = 'none';
            alert('An error occurred while processing your image');
            resetForm();
        });
    }

    function resetForm() {
        uploadArea.style.display = 'block';
        resultArea.style.display = 'none';
        fileInput.value = '';
    }
});