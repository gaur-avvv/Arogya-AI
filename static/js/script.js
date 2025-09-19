// Arogya AI - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading, hide results
        showLoading();
        hideResults();
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            data[key] = value.trim();
        });
        
        // Validate required fields
        const requiredFields = ['symptoms', 'age', 'height', 'weight', 'gender'];
        const missingFields = requiredFields.filter(field => !data[field]);
        
        if (missingFields.length > 0) {
            hideLoading();
            showAlert('Please fill in all required fields: ' + missingFields.join(', '), 'danger');
            return;
        }
        
        try {
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (response.ok && result.success) {
                displayResults(result);
            } else {
                showAlert(result.error || 'Prediction failed. Please try again.', 'danger');
            }
            
        } catch (error) {
            hideLoading();
            showAlert('Network error. Please check your connection and try again.', 'danger');
            console.error('Error:', error);
        }
    });
    
    function showLoading() {
        loadingSection.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Scroll to loading section
        loadingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    function hideLoading() {
        loadingSection.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search me-2"></i>Get Prediction & Recommendations';
    }
    
    function showResults() {
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
    
    function hideResults() {
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('fade-in');
    }
    
    function displayResults(result) {
        const { prediction, ayurvedic_recommendations } = result;
        
        const html = `
            <!-- Prediction Summary -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="alert alert-success d-flex align-items-center">
                        <i class="fas fa-check-circle fa-2x me-3"></i>
                        <div>
                            <h5 class="mb-1">Prediction Complete!</h5>
                            <p class="mb-0">Your personalized Ayurvedic treatment plan is ready.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Main Prediction -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="result-card">
                        <h6><i class="fas fa-stethoscope me-2"></i>Medical Prediction</h6>
                        <div class="prediction-badge bg-primary text-white mb-2">
                            ${prediction.disease}
                        </div>
                        <p class="mb-0"><strong>Confidence:</strong> ${prediction.confidence}</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="result-card">
                        <h6><i class="fas fa-user me-2"></i>Your Constitution</h6>
                        <div class="prediction-badge bg-info text-white mb-2">
                            ${prediction.user_body_type}
                        </div>
                        <p class="mb-0"><strong>Body Type:</strong> Ayurvedic Constitution</p>
                    </div>
                </div>
            </div>
            
            <!-- Ayurvedic Recommendations -->
            <div class="row">
                <div class="col-12">
                    <h5 class="border-bottom pb-2 mb-4">
                        <i class="fas fa-leaf me-2"></i>Your Personalized Ayurvedic Treatment Plan
                    </h5>
                </div>
                
                <!-- Herbs -->
                <div class="col-md-6 mb-4">
                    <div class="result-card herbs-section">
                        <h6><i class="fas fa-seedling me-2"></i>Recommended Herbs</h6>
                        <div class="mb-3">
                            <strong class="sanskrit-text">Sanskrit Names:</strong><br>
                            ${ayurvedic_recommendations.herbs_sanskrit}
                        </div>
                        <div class="mb-3">
                            <strong class="english-text">English Names:</strong><br>
                            ${ayurvedic_recommendations.herbs_english}
                        </div>
                        <div>
                            <strong class="effects-text">Therapeutic Effects:</strong><br>
                            ${ayurvedic_recommendations.herbs_effects}
                        </div>
                    </div>
                </div>
                
                <!-- Therapies -->
                <div class="col-md-6 mb-4">
                    <div class="result-card therapies-section">
                        <h6><i class="fas fa-spa me-2"></i>Recommended Therapies</h6>
                        <div class="mb-3">
                            <strong class="sanskrit-text">Sanskrit Therapies:</strong><br>
                            ${ayurvedic_recommendations.therapies_sanskrit}
                        </div>
                        <div class="mb-3">
                            <strong class="english-text">English Therapies:</strong><br>
                            ${ayurvedic_recommendations.therapies_english}
                        </div>
                        <div>
                            <strong class="effects-text">Therapy Benefits:</strong><br>
                            ${ayurvedic_recommendations.therapies_effects}
                        </div>
                    </div>
                </div>
                
                <!-- Dietary Recommendations -->
                <div class="col-12 mb-4">
                    <div class="result-card diet-section">
                        <h6><i class="fas fa-utensils me-2"></i>Dietary Recommendations</h6>
                        <div class="diet-text">
                            ${ayurvedic_recommendations.dietary_recommendations}
                        </div>
                    </div>
                </div>
                
                <!-- Body Type Effects -->
                <div class="col-12 mb-4">
                    <div class="result-card effects-section">
                        <h6><i class="fas fa-user-cog me-2"></i>How This Treatment Benefits Your Body Type</h6>
                        <div class="effects-text">
                            ${ayurvedic_recommendations.body_type_effects}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Action Buttons -->
            <div class="row mt-4">
                <div class="col-12 text-center">
                    <button class="btn btn-outline-primary me-3" onclick="window.print()">
                        <i class="fas fa-print me-2"></i>Print Results
                    </button>
                    <button class="btn btn-outline-secondary" onclick="resetForm()">
                        <i class="fas fa-redo me-2"></i>New Assessment
                    </button>
                </div>
            </div>
        `;
        
        resultsContent.innerHTML = html;
        showResults();
    }
    
    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert-custom');
        existingAlerts.forEach(alert => alert.remove());
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-custom mt-3`;
        alertDiv.innerHTML = `
            <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        form.parentNode.insertBefore(alertDiv, form);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    // Global function for resetting form
    window.resetForm = function() {
        form.reset();
        hideResults();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    
    // Input validation and formatting
    const ageInput = document.getElementById('age');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    
    // Age validation
    ageInput.addEventListener('input', function() {
        const value = parseInt(this.value);
        if (value < 1) this.value = 1;
        if (value > 120) this.value = 120;
    });
    
    // Height validation
    heightInput.addEventListener('input', function() {
        const value = parseFloat(this.value);
        if (value < 50) this.value = 50;
        if (value > 250) this.value = 250;
    });
    
    // Weight validation
    weightInput.addEventListener('input', function() {
        const value = parseFloat(this.value);
        if (value < 10) this.value = 10;
        if (value > 300) this.value = 300;
    });
    
    // Auto-resize textarea
    const symptomsTextarea = document.getElementById('symptoms');
    symptomsTextarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });
    
    // Enhance form UX
    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentNode.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentNode.classList.remove('focused');
        });
    });
});