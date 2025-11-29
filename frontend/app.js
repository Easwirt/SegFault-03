// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const btnText = generateBtn.querySelector('.btn-text');
const btnLoader = generateBtn.querySelector('.btn-loader');
const chatContainer = document.getElementById('chat-container');
const newChatBtn = document.getElementById('new-chat-btn');

// State
let currentPlan = [];
let completedTasks = 0;
let conversationHistory = [];
let currentMessageElement = null;
let isGenerating = false;
let generatedFiles = [];

// Generate function
async function generate(event) {
    // Prevent any default behavior
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    const prompt = promptInput.value.trim();
    
    if (!prompt || isGenerating) {
        return false;
    }

    // Add user message to chat
    addUserMessage(prompt);
    conversationHistory.push({ role: 'user', content: prompt });
    
    // Clear input
    promptInput.value = '';
    
    // Show new chat button
    newChatBtn.classList.remove('hidden');
    
    // Create AI response container
    currentMessageElement = createAIMessageContainer();
    
    // Reset state for new generation
    currentPlan = [];
    completedTasks = 0;
    generatedFiles = [];
    
    setLoading(true);
    isGenerating = true;

    try {
        // Build the full prompt with conversation context
        const fullPrompt = buildContextualPrompt(prompt);
        
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('Request timeout - aborting');
            controller.abort();
        }, 300000); // 5 minute timeout
        
        const response = await fetch(`${API_BASE_URL}/generate?prompt=${encodeURIComponent(fullPrompt)}`, {
            method: 'POST',
            headers: {
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache',
            },
            signal: controller.signal,
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let lastEventTime = Date.now();

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                // Process any remaining buffer
                if (buffer.trim()) {
                    processSSEBuffer(buffer);
                }
                break;
            }

            // Update last event time
            lastEventTime = Date.now();

            // Append new data to buffer
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete SSE messages
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || ''; // Keep incomplete message in buffer
            
            for (const message of lines) {
                if (message.trim()) {
                    processSSEBuffer(message);
                }
            }
        }

    } catch (error) {
        console.error('Error:', error);
        showErrorInMessage('Error: ' + error.message);
    } finally {
        setLoading(false);
        isGenerating = false;
        scrollToBottom();
    }
}

// Process SSE buffer and extract data
function processSSEBuffer(message) {
    const lines = message.split('\n');
    for (const line of lines) {
        // Skip heartbeat comments
        if (line.startsWith(':')) {
            continue;
        }
        if (line.startsWith('data: ')) {
            try {
                const data = JSON.parse(line.slice(6));
                handleEvent(data);
            } catch (e) {
                console.error('Error parsing SSE data:', e, line);
            }
        }
    }
}

// Build prompt with conversation context
function buildContextualPrompt(newPrompt) {
    if (conversationHistory.length <= 1) {
        return newPrompt;
    }
    
    // Include previous context for follow-up questions
    let context = "Previous conversation:\n";
    for (let i = 0; i < conversationHistory.length - 1; i++) {
        const msg = conversationHistory[i];
        context += `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}\n`;
    }
    context += `\nNew request: ${newPrompt}`;
    
    return context;
}

// Add user message to chat
function addUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Create AI message container
function createAIMessageContainer() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ai-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="status-indicator">
                <span class="status-dot"></span>
                <span class="status-text">Initializing...</span>
            </div>
            <div class="plan-section hidden">
                <h4>📋 Plan</h4>
                <ul class="plan-list"></ul>
            </div>
            <div class="tasks-section hidden">
                <h4>⚡ Tasks</h4>
                <div class="tasks-container"></div>
            </div>
            <div class="files-section hidden">
                <h4>📁 Generated Files</h4>
                <div class="files-container"></div>
            </div>
            <div class="answer-section hidden">
                <div class="final-answer"></div>
            </div>
            <div class="tokens-section hidden">
                <div class="token-stream"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

// Handle SSE events
function handleEvent(data) {
    if (!currentMessageElement) return;
    
    switch (data.type) {
        case 'node_start':
            handleNodeStart(data.node);
            break;
        case 'plan':
            handlePlan(data.plan);
            break;
        case 'task_complete':
            // Pass task_index
            handleTaskComplete(data.task, data.result, data.task_index);
            break;
        case 'file_status':
            // New handler for file creation status
            handleFileStatus(data.task, data.status, data.task_index);
            break;
        case 'file_generated':
            // Legacy support - convert to file_status format
            handleFileStatus(data.task, {
                created: true,
                filename: data.filename,
                filepath: data.filepath,
                fileType: 'text',
                message: data.result,
                downloadUrl: `${API_BASE_URL}/download?filepath=${encodeURIComponent(data.filepath)}`,
                previewUrl: `${API_BASE_URL}/preview?filepath=${encodeURIComponent(data.filepath)}`
            }, data.task_index);
            break;
        case 'token':
            handleToken(data.content);
            break;
        case 'final_answer':
            handleFinalAnswer(data.content, data.files);
            break;
        case 'done':
            handleDone();
            break;
        case 'error':
            handleError(data.message);
            break;
    }
    scrollToBottom();
}

// Event handlers
function handleNodeStart(node) {
    const nodeNames = {
        'planner': '📋 Planning...',
        'executor': '⚡ Executing tasks...',
        'responder': '✨ Generating final answer...'
    };
    setMessageStatus(nodeNames[node] || `Running ${node}...`);
}

function handlePlan(plan) {
    currentPlan = plan;
    completedTasks = 0;
    
    const planSection = currentMessageElement.querySelector('.plan-section');
    const planList = currentMessageElement.querySelector('.plan-list');
    
    planSection.classList.remove('hidden');
    planList.innerHTML = '';
    
    plan.forEach((task, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="task-number">${index + 1}</span>
            <span class="task-text">${escapeHtml(task)}</span>
        `;
        li.dataset.index = index;
        planList.appendChild(li);
    });

    // Mark first task as active
    if (plan.length > 0) {
        planList.children[0].classList.add('active');
    }
}

function handleTaskComplete(task, result, taskIndex) {
    const tasksSection = currentMessageElement.querySelector('.tasks-section');
    const tasksContainer = currentMessageElement.querySelector('.tasks-container');
    const planList = currentMessageElement.querySelector('.plan-list');
    
    tasksSection.classList.remove('hidden');
    
    // Add task card
    const taskCard = document.createElement('div');
    taskCard.className = 'task-card';
    taskCard.innerHTML = `
        <h5>✓ ${escapeHtml(task)}</h5>
        <p>${escapeHtml(result)}</p>
    `;
    tasksContainer.appendChild(taskCard);

    // Update plan list using index if available
    let index = -1;
    if (typeof taskIndex === 'number') {
        index = taskIndex;
    } else {
        index = currentPlan.findIndex(t => t === task);
    }

    if (index !== -1 && planList.children[index]) {
        planList.children[index].classList.remove('active');
        planList.children[index].classList.add('completed');
        
        // Mark next task as active
        if (index + 1 < planList.children.length) {
            planList.children[index + 1].classList.add('active');
        }
    }

    completedTasks++;
    setMessageStatus(`Completed ${completedTasks}/${currentPlan.length} tasks...`);
}

function handleToken(content) {
    const tokensSection = currentMessageElement.querySelector('.tokens-section');
    const tokenStream = currentMessageElement.querySelector('.token-stream');
    
    tokensSection.classList.remove('hidden');
    tokenStream.textContent += content;
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle file creation status
function handleFileStatus(task, status, taskIndex) {
    const filesSection = currentMessageElement.querySelector('.files-section');
    const filesContainer = currentMessageElement.querySelector('.files-container');
    const tasksSection = currentMessageElement.querySelector('.tasks-section');
    const tasksContainer = currentMessageElement.querySelector('.tasks-container');
    const planList = currentMessageElement.querySelector('.plan-list');
    
    // Show files section
    filesSection.classList.remove('hidden');
    
    // Create file card with status
    const fileCard = document.createElement('div');
    fileCard.className = `file-card ${status.created ? 'file-created' : 'file-failed'}`;
    
    // Build status icon and message
    const statusIcon = status.created ? '✅' : '❌';
    const statusClass = status.created ? 'status-success' : 'status-error';
    const fileTypeIcon = getFileTypeIcon(status.fileType);
    
    let previewHtml = '';
    
    // Generate preview based on file type
    if (status.created && status.preview) {
        if (status.preview.type === 'text') {
            previewHtml = `
                <div class="file-preview text-preview">
                    <pre>${escapeHtml(status.preview.content)}${status.preview.truncated ? '\n...' : ''}</pre>
                </div>
            `;
        } else if (status.preview.type === 'image') {
            previewHtml = `
                <div class="file-preview image-preview">
                    <img src="${status.preview.dataUrl}" alt="${escapeHtml(status.filename)}" />
                </div>
            `;
        }
    }
    
    // Build action buttons
    let actionsHtml = '';
    if (status.created) {
        actionsHtml = `
            <div class="file-actions">
                <button class="download-btn" onclick="downloadFile('${escapeHtml(status.filepath)}', '${escapeHtml(status.filename)}')">
                    ⬇️ Download
                </button>
                ${status.previewUrl ? `<button class="preview-btn" onclick="openPreview('${escapeHtml(status.previewUrl)}')">
                    👁️ Preview
                </button>` : ''}
            </div>
        `;
    }
    
    fileCard.innerHTML = `
        <div class="file-header">
            <div class="file-info">
                <span class="file-icon">${fileTypeIcon}</span>
                <span class="file-name">${escapeHtml(status.filename)}</span>
                ${status.fileSize ? `<span class="file-size">(${formatFileSize(status.fileSize)})</span>` : ''}
            </div>
            <span class="file-status ${statusClass}">
                ${statusIcon} ${status.created ? 'Created' : 'Failed'}
            </span>
        </div>
        ${status.message ? `<p class="file-message">${escapeHtml(status.message)}</p>` : ''}
        ${previewHtml}
        ${actionsHtml}
    `;
    filesContainer.appendChild(fileCard);
    
    // Also add to tasks for tracking
    tasksSection.classList.remove('hidden');
    const taskCard = document.createElement('div');
    taskCard.className = `task-card ${status.created ? 'file-task' : 'file-task-failed'}`;
    taskCard.innerHTML = `
        <h5>${status.created ? '✓' : '✗'} ${escapeHtml(task)}</h5>
        <p>${status.created ? '📁' : '⚠️'} ${escapeHtml(status.message)}</p>
    `;
    tasksContainer.appendChild(taskCard);
    
    // Update plan list using index if available
    let index = -1;
    if (typeof taskIndex === 'number') {
        index = taskIndex;
    } else {
        index = currentPlan.findIndex(t => t === task);
    }

    if (index !== -1 && planList.children[index]) {
        planList.children[index].classList.remove('active');
        planList.children[index].classList.add(status.created ? 'completed' : 'failed');
        
        if (index + 1 < planList.children.length) {
            planList.children[index + 1].classList.add('active');
        }
    }
    
    completedTasks++;
    if (status.created) {
        generatedFiles.push({ 
            name: status.filename, 
            path: status.filepath,
            downloadUrl: status.downloadUrl,
            previewUrl: status.previewUrl
        });
    }
    setMessageStatus(`Completed ${completedTasks}/${currentPlan.length} tasks...`);
}

// Get icon based on file type
function getFileTypeIcon(fileType) {
    switch (fileType) {
        case 'text': return '📄';
        case 'image': return '🖼️';
        default: return '📁';
    }
}

// Open preview in new window/tab
function openPreview(url) {
    window.open(url, '_blank');
}

function handleFinalAnswer(content, files) {
    const answerSection = currentMessageElement.querySelector('.answer-section');
    const finalAnswer = currentMessageElement.querySelector('.final-answer');
    const tokensSection = currentMessageElement.querySelector('.tokens-section');
    const filesSection = currentMessageElement.querySelector('.files-section');
    const filesContainer = currentMessageElement.querySelector('.files-container');
    
    answerSection.classList.remove('hidden');
    finalAnswer.textContent = content;
    
    // Hide tokens section when we have final answer
    tokensSection.classList.add('hidden');
    
    // Show files if any were generated (from final answer data)
    if (files && files.length > 0) {
        filesSection.classList.remove('hidden');
        // Only add files that weren't already added
        files.forEach(file => {
            // Handle both old format (name/path) and new format (filename/filepath)
            const filename = file.filename || file.name;
            const filepath = file.filepath || file.path;
            const existing = generatedFiles.find(f => (f.path === filepath) || (f.filepath === filepath));
            if (!existing) {
                const fileCard = document.createElement('div');
                fileCard.className = `file-card ${file.created !== false ? 'file-created' : 'file-failed'}`;
                
                const statusIcon = file.created !== false ? '✅' : '❌';
                const statusText = file.created !== false ? 'Created' : 'Failed';
                const fileTypeIcon = getFileTypeIcon(file.fileType || 'text');
                
                let actionsHtml = '';
                if (file.created !== false) {
                    const downloadUrl = file.downloadUrl || `${API_BASE_URL}/download?filepath=${encodeURIComponent(filepath)}`;
                    const previewUrl = file.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(filepath)}`;
                    actionsHtml = `
                        <div class="file-actions">
                            <button class="download-btn" onclick="downloadFile('${escapeHtml(filepath)}', '${escapeHtml(filename)}')">
                                ⬇️ Download
                            </button>
                            <button class="preview-btn" onclick="openPreview('${escapeHtml(previewUrl)}')">
                                👁️ Preview
                            </button>
                        </div>
                    `;
                }
                
                fileCard.innerHTML = `
                    <div class="file-header">
                        <div class="file-info">
                            <span class="file-icon">${fileTypeIcon}</span>
                            <span class="file-name">${escapeHtml(filename)}</span>
                            ${file.fileSize ? `<span class="file-size">(${formatFileSize(file.fileSize)})</span>` : ''}
                        </div>
                        <span class="file-status ${file.created !== false ? 'status-success' : 'status-error'}">
                            ${statusIcon} ${statusText}
                        </span>
                    </div>
                    ${actionsHtml}
                `;
                filesContainer.appendChild(fileCard);
            }
        });
    }
    
    // Add to conversation history
    conversationHistory.push({ role: 'assistant', content: content });
}

function handleDone() {
    setMessageStatus('✓ Complete', 'done');
    
    // Collapse plan and tasks after completion
    setTimeout(() => {
        const planSection = currentMessageElement.querySelector('.plan-section');
        const tasksSection = currentMessageElement.querySelector('.tasks-section');
        planSection.classList.add('collapsed');
        tasksSection.classList.add('collapsed');
    }, 1000);
}

function handleError(message) {
    setMessageStatus('Error: ' + message, 'error');
}

function showErrorInMessage(message) {
    if (currentMessageElement) {
        setMessageStatus(message, 'error');
    }
}

// UI helpers
function setLoading(loading) {
    generateBtn.disabled = loading;
    btnText.textContent = loading ? 'Generating...' : 'Send';
    btnLoader.classList.toggle('hidden', !loading);
}

function setMessageStatus(text, state = '') {
    if (!currentMessageElement) return;
    
    const statusIndicator = currentMessageElement.querySelector('.status-indicator');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');
    
    statusText.textContent = text;
    statusDot.classList.remove('done', 'error');
    if (state) {
        statusDot.classList.add(state);
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Download file function
async function downloadFile(filepath, filename) {
    try {
        const response = await fetch(`${API_BASE_URL}/download?filepath=${encodeURIComponent(filepath)}`);
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error('Download error:', error);
        alert('Failed to download file: ' + error.message);
    }
}

// Start new chat
function startNewChat(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    chatContainer.innerHTML = '';
    conversationHistory = [];
    currentMessageElement = null;
    newChatBtn.classList.add('hidden');
    promptInput.focus();
    return false;
}

// ==========================================
// MCP Server Management
// ==========================================

const mcpBtn = document.getElementById('mcp-btn');
const mcpModal = document.getElementById('mcp-modal');
const mcpModalClose = document.getElementById('mcp-modal-close');
const mcpModalOverlay = mcpModal.querySelector('.modal-overlay');
const mcpNameInput = document.getElementById('mcp-name');
const mcpUrlInput = document.getElementById('mcp-url');
const mcpDescriptionInput = document.getElementById('mcp-description');
const mcpTestBtn = document.getElementById('mcp-test-btn');
const mcpAddBtn = document.getElementById('mcp-add-btn');
const mcpAddStatus = document.getElementById('mcp-add-status');
const mcpServersContainer = document.getElementById('mcp-servers-container');

let mcpServers = [];

// Open MCP modal
function openMcpModal() {
    mcpModal.classList.remove('hidden');
    loadMcpServers();
}

// Close MCP modal
function closeMcpModal() {
    mcpModal.classList.add('hidden');
    clearMcpForm();
}

// Clear form
function clearMcpForm() {
    mcpNameInput.value = '';
    mcpUrlInput.value = '';
    mcpDescriptionInput.value = '';
    hideMcpStatus();
}

// Show status message
function showMcpStatus(message, type = 'loading') {
    mcpAddStatus.textContent = message;
    mcpAddStatus.className = `mcp-status ${type}`;
    mcpAddStatus.classList.remove('hidden');
}

// Hide status message
function hideMcpStatus() {
    mcpAddStatus.classList.add('hidden');
}

// Load MCP servers list
async function loadMcpServers() {
    try {
        const response = await fetch(`${API_BASE_URL}/mcp/servers`);
        const data = await response.json();
        
        if (data.success) {
            mcpServers = data.servers;
            renderMcpServers();
            updateMcpButton();
        }
    } catch (error) {
        console.error('Failed to load MCP servers:', error);
    }
}

// Render servers list
function renderMcpServers() {
    if (mcpServers.length === 0) {
        mcpServersContainer.innerHTML = '<p class="no-servers">No MCP servers connected</p>';
        return;
    }
    
    mcpServersContainer.innerHTML = mcpServers.map(server => `
        <div class="server-card" data-name="${escapeHtml(server.name)}">
            <div class="server-card-header">
                <h4>
                    <span class="server-status-dot ${server.connected ? 'connected' : 'disconnected'}"></span>
                    ${escapeHtml(server.name)}
                </h4>
                <button class="server-remove-btn" onclick="removeMcpServer('${escapeHtml(server.name)}')" title="Remove server">
                    🗑️
                </button>
            </div>
            <div class="server-url">${escapeHtml(server.url)}</div>
            ${server.description ? `<div class="server-description">${escapeHtml(server.description)}</div>` : ''}
            ${server.error ? `<div class="server-error">⚠️ ${escapeHtml(server.error)}</div>` : ''}
            ${server.tools && server.tools.length > 0 ? `
                <div class="server-tools">
                    <h5>Available Tools (${server.tools.length})</h5>
                    <div class="server-tools-list">
                        ${server.tools.map(tool => `
                            <span class="tool-tag" title="${escapeHtml(tool.description || '')}">${escapeHtml(tool.name)}</span>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `).join('');
}

// Update MCP button state
function updateMcpButton() {
    if (mcpServers.length > 0) {
        mcpBtn.classList.add('has-servers');
        mcpBtn.textContent = `🔌 MCP (${mcpServers.length})`;
    } else {
        mcpBtn.classList.remove('has-servers');
        mcpBtn.textContent = '🔌 MCP';
    }
}

// Test MCP connection
async function testMcpConnection() {
    const name = mcpNameInput.value.trim();
    const url = mcpUrlInput.value.trim();
    const description = mcpDescriptionInput.value.trim();
    
    if (!name || !url) {
        showMcpStatus('Please enter a name and URL', 'error');
        return;
    }
    
    showMcpStatus('Testing connection...', 'loading');
    mcpTestBtn.disabled = true;
    mcpAddBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}/mcp/test-connection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, url, description })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMcpStatus(`✅ Connection successful! Found ${data.tools_discovered} tools.`, 'success');
        } else {
            showMcpStatus(`❌ Connection failed: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showMcpStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        mcpTestBtn.disabled = false;
        mcpAddBtn.disabled = false;
    }
}

// Add MCP server
async function addMcpServer() {
    const name = mcpNameInput.value.trim();
    const url = mcpUrlInput.value.trim();
    const description = mcpDescriptionInput.value.trim();
    
    if (!name || !url) {
        showMcpStatus('Please enter a name and URL', 'error');
        return;
    }
    
    // Check for duplicate name
    if (mcpServers.some(s => s.name === name)) {
        showMcpStatus('A server with this name already exists', 'error');
        return;
    }
    
    showMcpStatus('Adding server...', 'loading');
    mcpTestBtn.disabled = true;
    mcpAddBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}/mcp/servers/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, url, description })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMcpStatus(`✅ ${data.message}`, 'success');
            clearMcpForm();
            await loadMcpServers();
        } else {
            showMcpStatus(`❌ ${data.message || 'Failed to add server'}`, 'error');
        }
    } catch (error) {
        showMcpStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        mcpTestBtn.disabled = false;
        mcpAddBtn.disabled = false;
    }
}

// Remove MCP server
async function removeMcpServer(name) {
    if (!confirm(`Remove MCP server "${name}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/mcp/servers/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        
        const data = await response.json();
        
        if (data.success) {
            await loadMcpServers();
        } else {
            alert(`Failed to remove server: ${data.message || 'Unknown error'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// MCP event listeners
mcpBtn.addEventListener('click', openMcpModal);
mcpModalClose.addEventListener('click', closeMcpModal);
mcpModalOverlay.addEventListener('click', closeMcpModal);
mcpTestBtn.addEventListener('click', testMcpConnection);
mcpAddBtn.addEventListener('click', addMcpServer);

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !mcpModal.classList.contains('hidden')) {
        closeMcpModal();
    }
});

// Load MCP servers on page load
window.addEventListener('load', () => {
    loadMcpServers();
});

// ==========================================
// End MCP Server Management
// ==========================================

// Enter key support
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        generate(e);
    }
});

// Button click handler
generateBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    generate(e);
});

// New chat button handler
newChatBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    startNewChat(e);
});

// Prevent form submission
document.getElementById('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    e.stopPropagation();
    return false;
});

// Focus input on load
window.addEventListener('load', () => {
    promptInput.focus();
});
