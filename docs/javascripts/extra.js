// SpaRRTa - Custom JavaScript for Interactive Features

document.addEventListener('DOMContentLoaded', function() {
  // Initialize all interactive features
  initCitationCopy();
  initScrollAnimations();
  initImageZoom();
  initSceneVizSlider();
  initPredictionDemo();
});

/**
 * Citation Copy Functionality
 */
function initCitationCopy() {
  const copyButtons = document.querySelectorAll('.citation-copy-btn');
  
  copyButtons.forEach(button => {
    button.addEventListener('click', function() {
      const citationBox = this.closest('.citation-box');
      const codeBlock = citationBox.querySelector('pre code');
      
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.textContent).then(() => {
          const originalText = this.textContent;
          this.textContent = 'Copied!';
          this.style.background = '#4caf50';
          
          setTimeout(() => {
            this.textContent = originalText;
            this.style.background = '';
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy:', err);
        });
      }
    });
  });
}

/**
 * Scroll-triggered Animations
 */
function initScrollAnimations() {
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe elements with animation classes
  document.querySelectorAll('.feature-card, .author-card, .env-card, .stat-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}

// Add animation class
document.addEventListener('DOMContentLoaded', function() {
  const style = document.createElement('style');
  style.textContent = `
    .animate-in {
      opacity: 1 !important;
      transform: translateY(0) !important;
    }
  `;
  document.head.appendChild(style);
});

/**
 * Image Zoom Enhancement
 */
function initImageZoom() {
  // Add click-to-zoom for teaser images
  document.querySelectorAll('.teaser-image').forEach(img => {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', function() {
      if (typeof GLightbox !== 'undefined') {
        const lightbox = GLightbox({
          elements: [{ href: this.src, type: 'image' }],
          touchNavigation: true,
          loop: false,
          autoplayVideos: true
        });
        lightbox.open();
      }
    });
  });
}

/**
 * Smooth Scroll for Anchor Links
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const targetId = this.getAttribute('href');
    if (targetId === '#') return;
    
    const targetElement = document.querySelector(targetId);
    if (targetElement) {
      e.preventDefault();
      targetElement.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

/**
 * Tab Synchronization
 * Keeps tabs in sync across the page
 */
document.addEventListener('DOMContentLoaded', function() {
  const tabGroups = {};
  
  document.querySelectorAll('.tabbed-set').forEach(tabSet => {
    const labels = tabSet.querySelectorAll('.tabbed-labels label');
    labels.forEach(label => {
      label.addEventListener('click', function() {
        const tabName = this.textContent.trim();
        // Sync other tab sets with the same tab
        document.querySelectorAll('.tabbed-set').forEach(otherSet => {
          if (otherSet !== tabSet) {
            const matchingLabel = Array.from(otherSet.querySelectorAll('.tabbed-labels label'))
              .find(l => l.textContent.trim() === tabName);
            if (matchingLabel) {
              matchingLabel.click();
            }
          }
        });
      });
    });
  });
});

/**
 * Results Table Sorting (if needed)
 */
function sortTable(tableId, columnIndex, ascending = true) {
  const table = document.getElementById(tableId);
  if (!table) return;
  
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  
  rows.sort((a, b) => {
    const aValue = a.cells[columnIndex].textContent.trim();
    const bValue = b.cells[columnIndex].textContent.trim();
    
    // Try numeric comparison first
    const aNum = parseFloat(aValue);
    const bNum = parseFloat(bValue);
    
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return ascending ? aNum - bNum : bNum - aNum;
    }
    
    // Fall back to string comparison
    return ascending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
  });
  
  rows.forEach(row => tbody.appendChild(row));
}

/**
 * Lazy Loading for Images
 */
document.addEventListener('DOMContentLoaded', function() {
  if ('IntersectionObserver' in window) {
    const lazyImages = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.removeAttribute('data-src');
          imageObserver.unobserve(img);
        }
      });
    });
    
    lazyImages.forEach(img => imageObserver.observe(img));
  }
});

/**
 * Scene Visualization Slider
 */
function initSceneVizSlider() {
  const container = document.querySelector('.scene-viz-container');
  if (!container) return;
  
  const totalImages = 60;
  let currentIndex = 0;
  let isDragging = false;
  
  const imageEl = document.getElementById('scene-viz-image');
  const thumbEl = document.getElementById('scene-viz-thumb');
  const fillEl = container.querySelector('.scene-viz-fill');
  const trackEl = container.querySelector('.scene-viz-track');
  const prevBtn = container.querySelector('.scene-viz-btn-prev');
  const nextBtn = container.querySelector('.scene-viz-btn-next');
  const currentEl = container.querySelector('.scene-viz-current');
  
  // Format image number with leading zeros
  function formatImageNumber(num) {
    return String(num).padStart(4, '0');
  }
  
  // Get base path from initial image src attribute
  const initialSrc = imageEl.getAttribute('src') || '';
  const basePath = initialSrc.replace(/sparrta_environment_viz_\d+\.png$/, '');
  
  // Update image
  function updateImage(index) {
    currentIndex = Math.max(0, Math.min(index, totalImages - 1));
    const imageNum = formatImageNumber(currentIndex);
    imageEl.src = `${basePath}sparrta_environment_viz_${imageNum}.png`;
    imageEl.alt = `Scene Visualization ${currentIndex + 1}`;
    
    // Update counter
    currentEl.textContent = currentIndex + 1;
    
    // Update slider position
    const percentage = (currentIndex / (totalImages - 1)) * 100;
    thumbEl.style.left = `${percentage}%`;
    fillEl.style.width = `${percentage}%`;
    
    // Update button states
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === totalImages - 1;
  }
  
  // Navigate to specific image
  function goToImage(index) {
    updateImage(index);
  }
  
  // Previous/Next navigation
  prevBtn.addEventListener('click', () => {
    if (currentIndex > 0) {
      updateImage(currentIndex - 1);
    }
  });
  
  nextBtn.addEventListener('click', () => {
    if (currentIndex < totalImages - 1) {
      updateImage(currentIndex + 1);
    }
  });
  
  // Slider track click
  trackEl.addEventListener('click', (e) => {
    if (isDragging) return;
    const rect = trackEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    const index = Math.round((percentage / 100) * (totalImages - 1));
    updateImage(index);
  });
  
  // Drag functionality
  function startDrag(e) {
    isDragging = true;
    trackEl.style.cursor = 'grabbing';
    thumbEl.style.cursor = 'grabbing';
    updateSliderPosition(e);
    
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', stopDrag);
    document.addEventListener('touchmove', onDrag, { passive: false });
    document.addEventListener('touchend', stopDrag);
    
    e.preventDefault();
  }
  
  function onDrag(e) {
    if (!isDragging) return;
    updateSliderPosition(e);
    e.preventDefault();
  }
  
  function stopDrag() {
    isDragging = false;
    trackEl.style.cursor = 'pointer';
    thumbEl.style.cursor = 'grab';
    
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', stopDrag);
    document.removeEventListener('touchmove', onDrag);
    document.removeEventListener('touchend', stopDrag);
  }
  
  function updateSliderPosition(e) {
    const rect = trackEl.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    const index = Math.round((percentage / 100) * (totalImages - 1));
    
    thumbEl.style.left = `${percentage}%`;
    fillEl.style.width = `${percentage}%`;
    
    if (index !== currentIndex) {
      updateImage(index);
    }
  }
  
  // Mouse drag
  thumbEl.addEventListener('mousedown', startDrag);
  trackEl.addEventListener('mousedown', (e) => {
    if (e.target === trackEl || e.target === fillEl) {
      startDrag(e);
    }
  });
  
  // Touch drag
  thumbEl.addEventListener('touchstart', startDrag, { passive: false });
  trackEl.addEventListener('touchstart', (e) => {
    if (e.target === trackEl || e.target === fillEl) {
      startDrag(e);
    }
  }, { passive: false });
  
  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (!container.contains(document.activeElement) && 
        !container.matches(':hover')) {
      return;
    }
    
    if (e.key === 'ArrowLeft' && currentIndex > 0) {
      e.preventDefault();
      updateImage(currentIndex - 1);
    } else if (e.key === 'ArrowRight' && currentIndex < totalImages - 1) {
      e.preventDefault();
      updateImage(currentIndex + 1);
    }
  });
  
  // Initialize
  updateImage(0);
}

/**
 * Prediction Demo
 */
function initPredictionDemo() {
  const container = document.querySelector('.prediction-demo-container');
  if (!container) return;
  
  // Prediction data from matching.yaml
  const predictionData = {
    'Bridge': {
      original: '../imgs/predictions/Bridge/original.jpg',
      camera: '../imgs/predictions/Bridge/blended_camera.jpg',
      human: '../imgs/predictions/Bridge/blended_human.jpg',
      camera_reference: 'Tree',
      camera_target: 'Car',
      human_reference: 'Tree',
      human_target: 'Car'
    },
    'Forest': {
      original: '../imgs/predictions/Forest/original.jpg',
      camera: '../imgs/predictions/Forest/blended_camera.jpg',
      human: '../imgs/predictions/Forest/blended_human.jpg',
      camera_reference: 'Tree',
      camera_target: 'Human',
      human_reference: 'Rock',
      human_target: 'Tree'
    },
    'City': {
      original: '../imgs/predictions/City/original.jpg',
      camera: '../imgs/predictions/City/blended_camera.jpg',
      human: '../imgs/predictions/City/blended_human.jpg',
      camera_reference: 'Traffic Cone',
      camera_target: 'Motorcycle',
      human_reference: 'Traffic Cone',
      human_target: 'Motorcycle'
    },
    'Desert': {
      original: '../imgs/predictions/Desert/original.jpg',
      camera: '../imgs/predictions/Desert/blended_camera.jpg',
      human: '../imgs/predictions/Desert/blended_human.jpg',
      camera_reference: 'Rock',
      camera_target: 'Car',
      human_reference: 'Rock',
      human_target: 'Car'
    },
    'Winter_Town': {
      original: '../imgs/predictions/Winter_Town/original.jpg',
      camera: '../imgs/predictions/Winter_Town/blended_camera.jpg',
      human: '../imgs/predictions/Winter_Town/blended_human.jpg',
      camera_reference: 'Snowman',
      camera_target: 'Husky',
      human_reference: 'Snowman',
      human_target: 'Husky'
    }
  };
  
  const envSelect = document.getElementById('prediction-env-select');
  const viewpointSelect = document.getElementById('prediction-viewpoint-select');
  const predictBtn = document.getElementById('prediction-predict-btn');
  const originalImg = document.getElementById('prediction-original-img');
  const predictionImg = document.getElementById('prediction-prediction-img');
  const predictionWrapper = document.getElementById('prediction-prediction-wrapper');
  const imagesContainer = document.getElementById('prediction-images-container');
  const referenceSpan = document.getElementById('prediction-reference');
  const targetSpan = document.getElementById('prediction-target');
  const tooltip = document.getElementById('prediction-tooltip');
  const tooltipPred = document.getElementById('prediction-tooltip-pred');
  const originalContainer = document.getElementById('prediction-original-container');
  const predictionContainer = document.getElementById('prediction-prediction-container');
  
  let currentEnv = envSelect.value;
  let currentViewpoint = viewpointSelect.value;
  
  // Update original image and info when environment/viewpoint changes
  function updateOriginalImage() {
    const data = predictionData[currentEnv];
    if (!data) return;
    
    originalImg.src = data.original;
    
    // Update reference and target based on viewpoint
    if (currentViewpoint === 'camera') {
      referenceSpan.textContent = data.camera_reference;
      targetSpan.textContent = data.camera_target;
    } else {
      referenceSpan.textContent = data.human_reference;
      targetSpan.textContent = data.human_target;
    }
    
    // Hide prediction when changing settings
    predictionWrapper.style.display = 'none';
    predictionWrapper.style.opacity = '0';
    imagesContainer.classList.remove('show-prediction');
    
    // Reset button state
    predictBtn.disabled = false;
    predictBtn.classList.remove('predicting');
    const btnIcon = predictBtn.querySelector('.btn-icon');
    if (!btnIcon || btnIcon.textContent !== '🔮') {
      predictBtn.innerHTML = '<span class="btn-icon">🔮</span> Predict';
    }
    originalContainer.classList.remove('predicting');
  }
  
  // Show prediction with loading state
  function showPrediction() {
    const data = predictionData[currentEnv];
    if (!data) return;
    
    // Store original button text
    const originalBtnText = '<span class="btn-icon">🔮</span> Predict';
    
    // Set loading state
    predictBtn.disabled = true;
    predictBtn.classList.add('predicting');
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Predicting...';
    
    // Add loading animation to original image
    originalContainer.classList.add('predicting');
    
    // Wait 1 second before showing prediction
    setTimeout(() => {
      const predictionFile = currentViewpoint === 'camera' ? data.camera : data.human;
      predictionImg.src = predictionFile;
      
      // Fade in prediction
      predictionWrapper.style.display = 'flex';
      predictionWrapper.style.opacity = '0';
      imagesContainer.classList.add('show-prediction');
      
      // Fade in animation
      setTimeout(() => {
        predictionWrapper.style.transition = 'opacity 0.5s ease';
        predictionWrapper.style.opacity = '1';
      }, 10);
      
      // Remove loading state
      predictBtn.disabled = false;
      predictBtn.classList.remove('predicting');
      predictBtn.innerHTML = originalBtnText;
      originalContainer.classList.remove('predicting');
    }, 1000);
  }
  
  // Tooltip functionality - setup once
  let tooltipReference = '';
  let tooltipTarget = '';
  
  function updateTooltipContent() {
    const data = predictionData[currentEnv];
    if (!data) return;
    
    tooltipReference = currentViewpoint === 'camera' ? data.camera_reference : data.human_reference;
    tooltipTarget = currentViewpoint === 'camera' ? data.camera_target : data.human_target;
  }
  
  // Setup tooltip handlers (only once)
  function setupTooltipHandlers(container, tooltipEl) {
    container.addEventListener('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Show tooltip with reference and target info
      tooltipEl.innerHTML = `
        <strong>Reference:</strong> ${tooltipReference}<br>
        <strong>Target:</strong> ${tooltipTarget}
      `;
      tooltipEl.style.left = `${x + 10}px`;
      tooltipEl.style.top = `${y + 10}px`;
      tooltipEl.classList.add('visible');
    });
    
    container.addEventListener('mouseleave', () => {
      tooltipEl.classList.remove('visible');
    });
  }
  
  // Initialize tooltips once
  setupTooltipHandlers(originalContainer, tooltip);
  setupTooltipHandlers(predictionContainer, tooltipPred);
  
  // Event listeners
  envSelect.addEventListener('change', (e) => {
    currentEnv = e.target.value;
    updateOriginalImage();
    updateTooltipContent();
  });
  
  viewpointSelect.addEventListener('change', (e) => {
    currentViewpoint = e.target.value;
    updateOriginalImage();
    updateTooltipContent();
  });
  
  predictBtn.addEventListener('click', () => {
    showPrediction();
  });
  
  // Initialize
  updateOriginalImage();
  updateTooltipContent();
}

/**
 * Console Easter Egg
 */
console.log(`
%c SpaRRTa %c Spatial Relation Recognition Task
%c Evaluating Spatial Intelligence in Visual Foundation Models

🔬 Built with Unreal Engine 5
📊 Benchmarking VFMs since 2024

`, 
'background: linear-gradient(135deg, #7c4dff, #536dfe); color: white; padding: 10px 20px; font-size: 20px; font-weight: bold; border-radius: 5px 0 0 5px;',
'background: #1a1a2e; color: #b47cff; padding: 10px 20px; font-size: 14px; border-radius: 0 5px 5px 0;',
'color: #9aa0a6; font-size: 12px;'
);

