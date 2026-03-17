// Chat interface with SSE streaming

const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const loadingIndicator = document.getElementById('loading');

// Conversation history — last 5 pairs (10 messages), client-side only
let conversationHistory = [];
const MAX_HISTORY_PAIRS = 5;

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add a message bubble to the chat and return its content element
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return contentDiv;
}

// Format restaurant cards as HTML
function formatRestaurants(restaurants) {
    if (!restaurants || restaurants.length === 0) return '';

    let html = '<div class="restaurants-list">';
    restaurants.forEach(function (restaurant) {
        html += '<div class="restaurant-card">';
        html += '<h4>' + escapeHtml(restaurant.name) + '</h4>';
        html += '<p><strong>Address:</strong> ' + escapeHtml(restaurant.address) + '</p>';
        html += '<p class="cuisine"><strong>Cuisine:</strong> ' + restaurant.cuisine.map(function (c) { return escapeHtml(c); }).join(', ') + '</p>';
        html += '<p><strong>Price:</strong> ' + escapeHtml(restaurant.price_range);
        if (restaurant.price_range_text) {
            html += ' (' + escapeHtml(restaurant.price_range_text) + ')';
        }
        html += '</p>';

        if (restaurant.rating) {
            html += '<p class="rating"><strong>Rating:</strong> ' + restaurant.rating + '/5.0 (' + restaurant.review_count + ' reviews)</p>';
        }
        if (restaurant.phone) {
            html += '<p><strong>Phone:</strong> ' + escapeHtml(restaurant.phone) + '</p>';
        }
        if (restaurant.website) {
            html += '<p><strong>Website:</strong> <a href="' + escapeHtml(restaurant.website) + '" target="_blank" rel="noopener">Visit</a></p>';
        }
        if (restaurant.features && restaurant.features.length > 0) {
            html += '<p><strong>Features:</strong> ' + restaurant.features.map(function (f) { return escapeHtml(f); }).join(', ') + '</p>';
        }
        if (restaurant.menu_url) {
            html += '<p><strong>Menu:</strong> <a href="' + escapeHtml(restaurant.menu_url) + '" target="_blank" rel="noopener">View full menu</a></p>';
        }
        html += '</div>';
    });
    html += '</div>';
    return html;
}

// Parse SSE events from a text chunk (may contain multiple events)
function parseSSEEvents(text) {
    var events = [];
    var lines = text.split('\n');
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('data: ')) {
            try {
                events.push(JSON.parse(lines[i].slice(6)));
            } catch (e) {
                // Skip malformed events
            }
        }
    }
    return events;
}

// Handle form submission with streaming
chatForm.addEventListener('submit', async function (e) {
    e.preventDefault();

    var message = userInput.value.trim();
    if (!message) return;

    // Show user message
    addMessage('<p>' + escapeHtml(message) + '</p>', 'user');

    // Clear input, disable send
    userInput.value = '';
    sendButton.disabled = true;
    loadingIndicator.style.display = 'block';

    // Track assistant message state
    var assistantContent = null;
    var textContainer = null;
    var accumulatedText = '';
    var streamedRestaurants = [];

    // Push user message to history before sending
    conversationHistory.push({ role: 'user', content: message });

    try {
        var response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                conversation_history: conversationHistory.slice(-(MAX_HISTORY_PAIRS * 2)),
            }),
        });

        if (!response.ok) {
            throw new Error('Server error: ' + response.status);
        }

        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        while (true) {
            var result = await reader.read();
            if (result.done) break;

            buffer += decoder.decode(result.value, { stream: true });

            // Process complete SSE events (delimited by double newline)
            var parts = buffer.split('\n\n');
            // Keep the last part as incomplete buffer
            buffer = parts.pop();

            for (var i = 0; i < parts.length; i++) {
                var events = parseSSEEvents(parts[i] + '\n');
                for (var j = 0; j < events.length; j++) {
                    var event = events[j];

                    switch (event.type) {
                        case 'restaurants':
                            // Phase 1: Show restaurant cards immediately
                            loadingIndicator.style.display = 'none';
                            assistantContent = addMessage('', 'assistant');
                            assistantContent.innerHTML = formatRestaurants(event.data);
                            streamedRestaurants = event.data;
                            // Create container for the LLM narrative text
                            textContainer = document.createElement('div');
                            textContainer.className = 'response-text';
                            assistantContent.appendChild(textContainer);
                            break;

                        case 'status':
                            if (textContainer) {
                                textContainer.innerHTML = '<em>Generating response...</em>';
                            }
                            break;

                        case 'token':
                            if (textContainer) {
                                if (accumulatedText === '') {
                                    textContainer.innerHTML = '';
                                }
                                accumulatedText += event.data;
                                textContainer.innerHTML = '<p>' + escapeHtml(accumulatedText) + '</p>';
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                            break;

                        case 'done':
                            // Push assistant response to history
                            conversationHistory.push({
                                role: 'assistant',
                                content: accumulatedText,
                                restaurants: streamedRestaurants,
                            });
                            // Trim to max pairs
                            if (conversationHistory.length > MAX_HISTORY_PAIRS * 2) {
                                conversationHistory = conversationHistory.slice(-(MAX_HISTORY_PAIRS * 2));
                            }
                            break;

                        case 'error':
                            loadingIndicator.style.display = 'none';
                            if (!assistantContent) {
                                addMessage('<p>' + escapeHtml(event.data) + '</p>', 'assistant');
                            } else if (textContainer) {
                                textContainer.innerHTML = '<p>' + escapeHtml(event.data) + '</p>';
                            }
                            break;
                    }
                }
            }
        }

        // Process any remaining buffer
        if (buffer.trim()) {
            var remaining = parseSSEEvents(buffer);
            for (var k = 0; k < remaining.length; k++) {
                if (remaining[k].type === 'token' && textContainer) {
                    accumulatedText += remaining[k].data;
                    textContainer.innerHTML = '<p>' + escapeHtml(accumulatedText) + '</p>';
                }
            }
        }

        // If no events were received at all
        if (!assistantContent) {
            addMessage('<p>No response received. Please try again.</p>', 'assistant');
        }

    } catch (error) {
        console.error('Stream error:', error);
        loadingIndicator.style.display = 'none';
        if (!assistantContent) {
            addMessage(
                '<p>Sorry, I encountered an error: ' + escapeHtml(error.message) + '. Please try again.</p>',
                'assistant'
            );
        }
    } finally {
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
        userInput.focus();
    }
});

// Focus input on load
userInput.focus();
