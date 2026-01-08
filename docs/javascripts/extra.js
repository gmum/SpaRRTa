// SpaRRTa - Custom JavaScript for Interactive Features

document.addEventListener('DOMContentLoaded', function() {
  // Initialize all interactive features
  initCitationCopy();
  initScrollAnimations();
  initImageZoom();
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

