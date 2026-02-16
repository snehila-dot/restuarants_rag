// Chat interface JavaScript

const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const loadingIndicator = document.getElementById('loading');

// Add user message to chat
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format restaurant data as HTML
function formatRestaurants(restaurants) {
    if (!restaurants || restaurants.length === 0) {
        return '';
    }
    
    let html = '<div class="restaurants-list">';
    
    restaurants.forEach(restaurant => {
        html += `
            <div class="restaurant-card">
                <h4>${escapeHtml(restaurant.name)}</h4>
                <p><strong>Address:</strong> ${escapeHtml(restaurant.address)}</p>
                <p class="cuisine"><strong>Cuisine:</strong> ${restaurant.cuisine.join(', ')}</p>
                <p><strong>Price:</strong> ${escapeHtml(restaurant.price_range)}</p>
        `;
        
        if (restaurant.rating) {
            html += `<p class="rating"><strong>Rating:</strong> ${restaurant.rating}/5.0 (${restaurant.review_count} reviews)</p>`;
        }
        
        if (restaurant.phone) {
            html += `<p><strong>Phone:</strong> ${escapeHtml(restaurant.phone)}</p>`;
        }
        
        if (restaurant.website) {
            html += `<p><strong>Website:</strong> <a href="${escapeHtml(restaurant.website)}" target="_blank" rel="noopener">Visit</a></p>`;
        }
        
        if (restaurant.features && restaurant.features.length > 0) {
            html += `<p><strong>Features:</strong> ${restaurant.features.join(', ')}</p>`;
        }
        
        html += `</div>`;
    });
    
    html += '</div>';
    return html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = userInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(`<p>${escapeHtml(message)}</p>`, 'user');
    
    // Clear input and disable button
    userInput.value = '';
    sendButton.disabled = true;
    loadingIndicator.style.display = 'block';
    
    try {
        // Send request to API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get response');
        }
        
        const data = await response.json();
        
        // Format response
        let responseHtml = `<p>${escapeHtml(data.message)}</p>`;
        
        // Add restaurants if any
        if (data.restaurants && data.restaurants.length > 0) {
            responseHtml += formatRestaurants(data.restaurants);
        }
        
        // Add assistant message
        addMessage(responseHtml, 'assistant');
        
    } catch (error) {
        console.error('Error:', error);
        addMessage(
            `<p>Sorry, I encountered an error: ${escapeHtml(error.message)}. Please try again.</p>`,
            'assistant'
        );
    } finally {
        // Re-enable button and hide loading
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
        userInput.focus();
    }
});

// Focus input on load
userInput.focus();
