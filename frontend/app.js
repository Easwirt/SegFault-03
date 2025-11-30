// Configuration
// Use relative URL if served from same origin, otherwise use localhost:8000
const API_BASE_URL = window.location.port === '8000' ? '' : 'http://localhost:8000';

// DOM Elements
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const btnText = generateBtn.querySelector('.btn-text');
const btnLoader = generateBtn.querySelector('.btn-loader');
const chatContainer = document.getElementById('chat-container');
const newChatBtn = document.getElementById('new-chat-btn');
const modeSelect = document.getElementById('mode-select');
const modeHint = document.getElementById('mode-hint');

// Sidebar Elements
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
const showSidebarBtn = document.getElementById('show-sidebar-btn');
const projectSelect = document.getElementById('project-select');
const newProjectBtn = document.getElementById('new-project-btn');
const editProjectBtn = document.getElementById('edit-project-btn');
const deleteProjectBtn = document.getElementById('delete-project-btn');
const chatHistoryList = document.getElementById('chat-history-list');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// Project Modal Elements
const projectModal = document.getElementById('project-modal');
const projectModalTitle = document.getElementById('project-modal-title');
const projectModalClose = document.getElementById('project-modal-close');
const projectNameInput = document.getElementById('project-name');
const projectDescriptionInput = document.getElementById('project-description');
const projectCancelBtn = document.getElementById('project-cancel-btn');
const projectSaveBtn = document.getElementById('project-save-btn');

// Rename Chat Modal Elements
const renameChatModal = document.getElementById('rename-chat-modal');
const renameChatModalClose = document.getElementById('rename-chat-modal-close');
const chatTitleInput = document.getElementById('chat-title-input');
const renameChatCancelBtn = document.getElementById('rename-chat-cancel-btn');
const renameChatSaveBtn = document.getElementById('rename-chat-save-btn');

// State
let currentPlan = [];
let completedTasks = 0;
let conversationHistory = [];
let currentMessageElement = null;
let isGenerating = false;
let generatedFiles = [];

// Chat History & Projects State
let projects = [];
let currentProjectId = 'default';
let chatHistories = {};
let currentChatId = null;
let editingProjectId = null;
let renamingChatId = null;

// Initialize storage
function initStorage() {
    // Load projects
    const savedProjects = localStorage.getItem('ai_agent_projects');
    if (savedProjects) {
        projects = JSON.parse(savedProjects);
    } else {
        projects = [{ id: 'default', name: 'Default Project', description: '', createdAt: Date.now() }];
        saveProjects();
    }
    
    // Load chat histories
    const savedHistories = localStorage.getItem('ai_agent_chat_histories');
    if (savedHistories) {
        chatHistories = JSON.parse(savedHistories);
    }
    
    // Ensure default project has a chat array
    if (!chatHistories['default']) {
        chatHistories['default'] = [];
    }
    
    renderProjects();
    renderChatHistory();
}

// Save projects to localStorage
function saveProjects() {
    localStorage.setItem('ai_agent_projects', JSON.stringify(projects));
}

// Save chat histories to localStorage
function saveChatHistories() {
    localStorage.setItem('ai_agent_chat_histories', JSON.stringify(chatHistories));
}

// Generate unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Render projects dropdown
function renderProjects() {
    projectSelect.innerHTML = projects.map(p => 
        `<option value="${p.id}" ${p.id === currentProjectId ? 'selected' : ''}>${escapeHtml(p.name)}</option>`
    ).join('');
}

// Render chat history
function renderChatHistory() {
    const chats = chatHistories[currentProjectId] || [];
    
    if (chats.length === 0) {
        chatHistoryList.innerHTML = '<p class="no-chats">No chat history</p>';
        return;
    }
    
    // Sort by date descending
    const sortedChats = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);
    
    chatHistoryList.innerHTML = sortedChats.map(chat => {
        const date = new Date(chat.updatedAt).toLocaleDateString();
        const preview = chat.messages.length > 0 
            ? chat.messages[chat.messages.length - 1].content.substring(0, 50) + '...'
            : 'Empty chat';
        
        return `
            <div class="chat-history-item ${chat.id === currentChatId ? 'active' : ''}" 
                 data-chat-id="${chat.id}" 
                 onclick="loadChat('${chat.id}')">
                <div class="chat-item-title">${escapeHtml(chat.title)}</div>
                <div class="chat-item-preview">${escapeHtml(preview)}</div>
                <div class="chat-item-date">${date}</div>
                <div class="chat-item-actions">
                    <button class="chat-action-btn" onclick="event.stopPropagation(); openRenameChat('${chat.id}')" title="Rename">✏️</button>
                    <button class="chat-action-btn danger" onclick="event.stopPropagation(); deleteChat('${chat.id}')" title="Delete">🗑️</button>
                </div>
            </div>
        `;
    }).join('');
}

// Create new chat
function createNewChat() {
    const chatId = generateId();
    const chat = {
        id: chatId,
        title: 'New Chat',
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now()
    };
    
    if (!chatHistories[currentProjectId]) {
        chatHistories[currentProjectId] = [];
    }
    chatHistories[currentProjectId].push(chat);
    saveChatHistories();
    
    return chatId;
}

// Save current chat
function saveCurrentChat() {
    console.log('saveCurrentChat called, currentChatId:', currentChatId);
    if (!currentChatId) return;
    
    const chats = chatHistories[currentProjectId] || [];
    const chatIndex = chats.findIndex(c => c.id === currentChatId);
    
    if (chatIndex !== -1) {
        chats[chatIndex].messages = conversationHistory;
        chats[chatIndex].updatedAt = Date.now();
        
        // Auto-generate title from first user message if still default
        if (chats[chatIndex].title === 'New Chat' && conversationHistory.length > 0) {
            const firstUserMsg = conversationHistory.find(m => m.role === 'user');
            if (firstUserMsg) {
                chats[chatIndex].title = firstUserMsg.content.substring(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '');
            }
        }
        
        saveChatHistories();
        console.log('saveCurrentChat: About to render chat history');
        renderChatHistory();
        console.log('saveCurrentChat: Finished rendering chat history');
    }
}

// Load a chat
function loadChat(chatId) {
    console.log('loadChat called:', chatId, 'isGenerating:', isGenerating, 'currentChatId:', currentChatId);
    console.trace('loadChat stack trace');
    
    // Don't load a different chat while generating
    if (isGenerating) {
        console.log('Cannot load chat while generating');
        return;
    }
    
    // Don't reload the current chat
    if (chatId === currentChatId) {
        console.log('Already on this chat');
        return;
    }
    
    const chats = chatHistories[currentProjectId] || [];
    const chat = chats.find(c => c.id === chatId);
    
    if (!chat) {
        console.log('Chat not found:', chatId);
        return;
    }
    
    console.log('Loading chat with messages:', chat.messages.length);
    
    currentChatId = chatId;
    conversationHistory = [...chat.messages];
    
    // Render messages
    chatContainer.innerHTML = '';
    conversationHistory.forEach((msg, index) => {
        console.log('Rendering message:', index, msg.role, msg.files ? `with ${msg.files.length} files` : 'no files');
        if (msg.role === 'user') {
            addUserMessage(msg.content);
        } else {
            // Pass full message object to preserve files info
            addStoredAIMessage(msg);
        }
    });
    
    renderChatHistory();
}

// Add stored AI message (with files support for history)
function addStoredAIMessage(msg) {
    const content = typeof msg === 'string' ? msg : msg.content;
    const files = typeof msg === 'object' ? msg.files : null;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ai-message';
    
    // Build files section HTML if there are files
    let filesHtml = '';
    if (files && files.length > 0) {
        const fileCards = files.map(file => {
            const filename = file.filename || file.name;
            const filepath = file.filepath || file.path;
            const fileTypeIcon = getFileTypeIcon(file.fileType || 'text');
            const statusIcon = file.created !== false ? '✅' : '❌';
            const statusText = file.created !== false ? 'Created' : 'Failed';
            const statusClass = file.created !== false ? 'status-success' : 'status-error';
            
            // Generate preview HTML for images using previewUrl
            let previewHtml = '';
            if (file.created !== false && file.fileType === 'image' && file.previewUrl) {
                previewHtml = `
                    <div class="file-preview image-preview">
                        <img src="${file.previewUrl}" alt="${escapeHtml(filename)}" 
                             onerror="this.onerror=null; this.parentElement.innerHTML='<p class=\\'preview-error\\'>Image no longer available</p>'" />
                    </div>
                `;
            }
            
            let actionsHtml = '';
            if (file.created !== false && filepath) {
                actionsHtml = `
                    <div class="file-actions">
                        <button class="download-btn" type="button" data-action="download" data-filepath="${encodeURIComponent(filepath)}" data-filename="${encodeURIComponent(filename)}">
                            ⬇️ Download
                        </button>
                        ${file.previewUrl ? `<button class="preview-btn" type="button" data-action="preview" data-url="${encodeURIComponent(file.previewUrl)}" data-filepath="${encodeURIComponent(filepath)}" data-filename="${encodeURIComponent(filename)}">
                            👁️ Preview
                        </button>` : ''}
                    </div>
                `;
            }
            
            return `
                <div class="file-card ${file.created !== false ? 'file-created' : 'file-failed'}">
                    <div class="file-header">
                        <div class="file-info">
                            <span class="file-icon">${fileTypeIcon}</span>
                            <span class="file-name">${escapeHtml(filename)}</span>
                            ${file.fileSize ? `<span class="file-size">(${formatFileSize(file.fileSize)})</span>` : ''}
                        </div>
                        <span class="file-status ${statusClass}">
                            ${statusIcon} ${statusText}
                        </span>
                    </div>
                    ${previewHtml}
                    ${actionsHtml}
                </div>
            `;
        }).join('');
        
        filesHtml = `
            <div class="files-section">
                <h4>📁 Generated Files</h4>
                <div class="files-container">${fileCards}</div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${filesHtml}
            <div class="answer-section">
                <div class="final-answer">${escapeHtml(content)}</div>
            </div>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
}

// Delete a chat
function deleteChat(chatId) {
    if (!confirm('Delete this chat?')) return;
    
    const chats = chatHistories[currentProjectId] || [];
    const chatIndex = chats.findIndex(c => c.id === chatId);
    
    if (chatIndex !== -1) {
        chats.splice(chatIndex, 1);
        saveChatHistories();
        
        if (currentChatId === chatId) {
            startNewChat();
        }
        
        renderChatHistory();
    }
}

// Open rename chat modal
function openRenameChat(chatId) {
    renamingChatId = chatId;
    const chats = chatHistories[currentProjectId] || [];
    const chat = chats.find(c => c.id === chatId);
    
    if (chat) {
        chatTitleInput.value = chat.title;
        renameChatModal.classList.remove('hidden');
        chatTitleInput.focus();
    }
}

// Save renamed chat
function saveRenamedChat() {
    const title = chatTitleInput.value.trim();
    if (!title || !renamingChatId) return;
    
    const chats = chatHistories[currentProjectId] || [];
    const chat = chats.find(c => c.id === renamingChatId);
    
    if (chat) {
        chat.title = title;
        saveChatHistories();
        renderChatHistory();
    }
    
    closeRenameChatModal();
}

// Close rename chat modal
function closeRenameChatModal() {
    renameChatModal.classList.add('hidden');
    renamingChatId = null;
    chatTitleInput.value = '';
}

// Clear all history for current project
function clearAllHistory() {
    if (!confirm('Clear all chat history for this project?')) return;
    
    chatHistories[currentProjectId] = [];
    saveChatHistories();
    startNewChat();
    renderChatHistory();
}

// Project Management
function openProjectModal(editId = null) {
    editingProjectId = editId;
    
    if (editId) {
        const project = projects.find(p => p.id === editId);
        if (project) {
            projectModalTitle.textContent = '✏️ Edit Project';
            projectNameInput.value = project.name;
            projectDescriptionInput.value = project.description || '';
        }
    } else {
        projectModalTitle.textContent = '📁 New Project';
        projectNameInput.value = '';
        projectDescriptionInput.value = '';
    }
    
    projectModal.classList.remove('hidden');
    projectNameInput.focus();
}

function closeProjectModal() {
    projectModal.classList.add('hidden');
    editingProjectId = null;
    projectNameInput.value = '';
    projectDescriptionInput.value = '';
}

function saveProject() {
    const name = projectNameInput.value.trim();
    if (!name) {
        alert('Please enter a project name');
        return;
    }
    
    if (editingProjectId) {
        // Edit existing
        const project = projects.find(p => p.id === editingProjectId);
        if (project) {
            project.name = name;
            project.description = projectDescriptionInput.value.trim();
        }
    } else {
        // Create new
        const projectId = generateId();
        projects.push({
            id: projectId,
            name: name,
            description: projectDescriptionInput.value.trim(),
            createdAt: Date.now()
        });
        chatHistories[projectId] = [];
        currentProjectId = projectId;
    }
    
    saveProjects();
    saveChatHistories();
    renderProjects();
    renderChatHistory();
    closeProjectModal();
    startNewChat();
}

function deleteProject() {
    if (currentProjectId === 'default') {
        alert('Cannot delete the default project');
        return;
    }
    
    if (!confirm('Delete this project and all its chat history?')) return;
    
    const projectIndex = projects.findIndex(p => p.id === currentProjectId);
    if (projectIndex !== -1) {
        projects.splice(projectIndex, 1);
        delete chatHistories[currentProjectId];
        
        currentProjectId = 'default';
        saveProjects();
        saveChatHistories();
        renderProjects();
        renderChatHistory();
        startNewChat();
    }
}

// Sidebar toggle
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
    showSidebarBtn.classList.toggle('visible', sidebar.classList.contains('collapsed'));
}

function showSidebar() {
    sidebar.classList.remove('collapsed');
    sidebar.classList.add('open');
    document.body.classList.remove('sidebar-collapsed');
    showSidebarBtn.classList.remove('visible');
}

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

    // Create new chat if needed - do this BEFORE setting isGenerating
    if (!currentChatId) {
        currentChatId = createNewChat();
        renderChatHistory(); // Update sidebar to show new chat
    }

    // Set generating state AFTER chat is created
    isGenerating = true;

    // Add user message to chat
    addUserMessage(prompt);
    conversationHistory.push({ role: 'user', content: prompt });
    
    // Save chat immediately
    saveCurrentChat();
    
    // Clear input
    promptInput.value = '';
    
    // Create AI response container
    currentMessageElement = createAIMessageContainer();
    
    // Reset state for new generation
    currentPlan = [];
    completedTasks = 0;
    generatedFiles = [];
    finalAnswerReceived = false;
    
    setLoading(true);

    try {
        // Build the full prompt with conversation context
        const fullPrompt = buildContextualPrompt(prompt);
        
        // Get selected mode
        const selectedMode = modeSelect ? modeSelect.value : 'auto';
        
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('Request timeout - aborting');
            controller.abort();
        }, 300000); // 5 minute timeout
        
        const response = await fetch(`${API_BASE_URL}/generate?prompt=${encodeURIComponent(fullPrompt)}&mode=${selectedMode}`, {
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
        let streamEnded = false;

        while (true) {
            let readResult;
            try {
                readResult = await reader.read();
            } catch (readError) {
                // Stream read error - this is often just the server closing the connection
                console.log('Stream read ended:', readError.message);
                streamEnded = true;
                break;
            }
            
            const { done, value } = readResult;
            
            if (done) {
                console.log('SSE stream done');
                streamEnded = true;
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
        
        // After stream ends, ensure we save if we have files but no final answer was received
        if (streamEnded && generatedFiles.length > 0 && !finalAnswerReceived) {
            console.log('Stream ended gracefully but no final_answer received, saving files...');
            const fallbackContent = 'Task completed. Files have been generated.';
            const messageEntry = { 
                role: 'assistant', 
                content: fallbackContent,
                files: generatedFiles.map(f => ({
                    filename: f.filename || f.name,
                    filepath: f.filepath || f.path,
                    fileType: f.fileType || 'text',
                    fileSize: f.fileSize,
                    downloadUrl: f.downloadUrl || `${API_BASE_URL}/download?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                    previewUrl: f.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                    created: f.created !== false
                }))
            };
            conversationHistory.push(messageEntry);
            
            // Show in UI
            const answerSection = currentMessageElement?.querySelector('.answer-section');
            const finalAnswer = currentMessageElement?.querySelector('.final-answer');
            if (answerSection && finalAnswer) {
                answerSection.classList.remove('hidden');
                finalAnswer.textContent = fallbackContent;
            }
            
            saveCurrentChat();
            setMessageStatus('✓ Complete', 'done');
        }

    } catch (error) {
        console.error('Error:', error);
        
        // If we have generated files but got a stream error, still save them
        if (generatedFiles.length > 0 && !finalAnswerReceived) {
            console.log('Stream error but have generated files, saving...');
            const fallbackContent = 'Generation completed (stream ended early). Files have been created.';
            const messageEntry = { 
                role: 'assistant', 
                content: fallbackContent,
                files: generatedFiles.map(f => ({
                    filename: f.filename || f.name,
                    filepath: f.filepath || f.path,
                    fileType: f.fileType || 'text',
                    fileSize: f.fileSize,
                    downloadUrl: f.downloadUrl || `${API_BASE_URL}/download?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                    previewUrl: f.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                    created: f.created !== false
                }))
            };
            conversationHistory.push(messageEntry);
            
            // Show in UI
            const answerSection = currentMessageElement?.querySelector('.answer-section');
            const finalAnswer = currentMessageElement?.querySelector('.final-answer');
            if (answerSection && finalAnswer) {
                answerSection.classList.remove('hidden');
                finalAnswer.textContent = fallbackContent;
            }
            
            saveCurrentChat();
            setMessageStatus('✓ Complete (with warnings)', 'done');
        } else {
            showErrorInMessage('Error: ' + error.message);
        }
    } finally {
        setLoading(false);
        isGenerating = false;
        finalAnswerReceived = false;
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
        case 'mode':
            handleModeEvent(data.mode);
            break;
        case 'cache_hit':
            handleCacheHit(data.similarity, data.cached_request);
            break;
        case 'node_start':
            handleNodeStart(data.node);
            break;
        case 'plan':
            handlePlan(data.plan);
            break;
        case 'task_start':
            handleTaskStart(data.task, data.task_index);
            break;
        case 'task_complete':
            // Pass task_index
            handleTaskComplete(data.task, data.result, data.task_index);
            break;
        case 'file_status':
            // New handler for file creation status
            console.log('file_status event:', data);
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
            console.log('final_answer event:', { content: data.content?.substring(0, 100), files: data.files });
            handleFinalAnswer(data.content, data.files);
            break;
        case 'done':
            console.log('done event received');
            handleDone();
            break;
        case 'error':
            handleError(data.message);
            break;
    }
    scrollToBottom();
}

// Event handlers
function handleModeEvent(mode) {
    const modeLabels = {
        'simple': '⚡ Simple Mode',
        'agent': '🤖 Agent Mode',
        'cached': '💾 Cached Response'
    };
    setMessageStatus(modeLabels[mode] || mode);
    
    // Add mode indicator to message
    const statusEl = currentMessageElement.querySelector('.message-status');
    if (statusEl) {
        statusEl.classList.add(`mode-${mode}`);
    }
}

function handleCacheHit(similarity, cachedRequest) {
    const percent = Math.round(similarity * 100);
    setMessageStatus(`💾 Cache hit (${percent}% similar)`);
    
    // Add cache indicator to message
    const statusEl = currentMessageElement.querySelector('.message-status');
    if (statusEl) {
        statusEl.classList.add('mode-cached');
        statusEl.title = `Similar to: "${cachedRequest?.substring(0, 50)}..."`;
    }
}

function handleTaskStart(task, taskIndex) {
    setMessageStatus(`⚙️ ${task}...`);
}

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
            // Check if we have inline data or need to use URL
            if (status.preview.dataUrl) {
                previewHtml = `
                    <div class="file-preview image-preview">
                        <img src="${status.preview.dataUrl}" alt="${escapeHtml(status.filename)}" />
                    </div>
                `;
            } else if (status.preview.useUrl && status.previewUrl) {
                // Use URL for large images
                previewHtml = `
                    <div class="file-preview image-preview">
                        <img src="${status.previewUrl}" alt="${escapeHtml(status.filename)}" 
                             onerror="this.onerror=null; this.parentElement.innerHTML='<p class=\\'preview-error\\'>Failed to load image preview</p>'" />
                    </div>
                `;
            }
        } else if (status.preview.type === 'error') {
            previewHtml = `
                <div class="file-preview error-preview">
                    <p>Preview error: ${escapeHtml(status.preview.message)}</p>
                </div>
            `;
        }
    } else if (status.created && status.fileType === 'image' && status.previewUrl) {
        // Fallback: if no preview object but it's an image, use previewUrl
        previewHtml = `
            <div class="file-preview image-preview">
                <img src="${status.previewUrl}" alt="${escapeHtml(status.filename)}" 
                     onerror="this.onerror=null; this.parentElement.innerHTML='<p class=\\'preview-error\\'>Failed to load image preview</p>'" />
            </div>
        `;
    }
    
    // Build action buttons
    let actionsHtml = '';
    if (status.created) {
        // Use data attributes for event delegation
        actionsHtml = `
            <div class="file-actions">
                <button class="download-btn" type="button" data-action="download" data-filepath="${encodeURIComponent(status.filepath)}" data-filename="${encodeURIComponent(status.filename)}">
                    ⬇️ Download
                </button>
                ${status.previewUrl ? `<button class="preview-btn" type="button" data-action="preview" data-url="${encodeURIComponent(status.previewUrl)}" data-filepath="${encodeURIComponent(status.filepath)}" data-filename="${encodeURIComponent(status.filename)}">
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
            filename: status.filename, 
            filepath: status.filepath,
            fileType: status.fileType,
            fileSize: status.fileSize,
            downloadUrl: status.downloadUrl,
            previewUrl: status.previewUrl,
            created: true
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

// Image Preview Modal Elements
const imagePreviewModal = document.getElementById('image-preview-modal');
const imagePreviewClose = document.getElementById('image-preview-close');
const imagePreviewImg = document.getElementById('image-preview-img');
const imagePreviewTitle = document.getElementById('image-preview-title');
const imagePreviewDownload = document.getElementById('image-preview-download');
let currentPreviewFilepath = '';
let currentPreviewFilename = '';

// Open image preview in modal
function openPreview(url, filename = 'image', filepath = '') {
    // Prevent any default navigation  
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    if (url) {
        imagePreviewImg.src = url;
        imagePreviewTitle.textContent = `🖼️ ${filename}`;
        currentPreviewFilepath = filepath;
        currentPreviewFilename = filename;
        imagePreviewModal.classList.remove('hidden');
    }
    return false;
}

// Close image preview modal
function closeImagePreview() {
    imagePreviewModal.classList.add('hidden');
    imagePreviewImg.src = '';
}

// Image preview modal event listeners
imagePreviewClose.addEventListener('click', closeImagePreview);
imagePreviewModal.querySelector('.modal-overlay').addEventListener('click', closeImagePreview);
imagePreviewDownload.addEventListener('click', () => {
    if (currentPreviewFilepath && currentPreviewFilename) {
        downloadFile(currentPreviewFilepath, currentPreviewFilename);
    }
});

// Track if final answer was received
let finalAnswerReceived = false;

function handleFinalAnswer(content, files) {
    console.log('handleFinalAnswer called:', { content: content?.substring(0, 50), files: files?.length || 0 });
    finalAnswerReceived = true;
    
    const answerSection = currentMessageElement.querySelector('.answer-section');
    const finalAnswer = currentMessageElement.querySelector('.final-answer');
    const tokensSection = currentMessageElement.querySelector('.tokens-section');
    const filesSection = currentMessageElement.querySelector('.files-section');
    const filesContainer = currentMessageElement.querySelector('.files-container');
    
    // Guard against null elements
    if (!answerSection || !finalAnswer) {
        console.error('handleFinalAnswer: Required elements not found');
        return;
    }
    
    answerSection.classList.remove('hidden');
    finalAnswer.textContent = content;
    
    // Hide tokens section when we have final answer
    if (tokensSection) {
        tokensSection.classList.add('hidden');
    }
    
    // Combine files from the response with our tracked generatedFiles
    let allFiles = [];
    if (files && files.length > 0) {
        allFiles = [...files];
    }
    // Add any files we tracked during execution that might not be in the final answer
    generatedFiles.forEach(gf => {
        const filepath = gf.filepath || gf.path;
        const exists = allFiles.find(f => (f.filepath === filepath) || (f.path === filepath));
        if (!exists) {
            allFiles.push(gf);
        }
    });
    
    // Show files if any were generated (from final answer data)
    if (allFiles.length > 0 && filesSection && filesContainer) {
        filesSection.classList.remove('hidden');
        // Only add files that weren't already added
        allFiles.forEach(file => {
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
                
                // Generate preview HTML for images
                let previewHtml = '';
                const previewUrl = file.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(filepath)}`;
                if (file.created !== false && file.fileType === 'image') {
                    if (file.preview && file.preview.dataUrl) {
                        previewHtml = `
                            <div class="file-preview image-preview">
                                <img src="${file.preview.dataUrl}" alt="${escapeHtml(filename)}" />
                            </div>
                        `;
                    } else {
                        previewHtml = `
                            <div class="file-preview image-preview">
                                <img src="${previewUrl}" alt="${escapeHtml(filename)}" 
                                     onerror="this.onerror=null; this.parentElement.innerHTML='<p class=\\'preview-error\\'>Failed to load image preview</p>'" />
                            </div>
                        `;
                    }
                }
                
                let actionsHtml = '';
                if (file.created !== false) {
                    actionsHtml = `
                        <div class="file-actions">
                            <button class="download-btn" type="button" data-action="download" data-filepath="${encodeURIComponent(filepath)}" data-filename="${encodeURIComponent(filename)}">
                                ⬇️ Download
                            </button>
                            <button class="preview-btn" type="button" data-action="preview" data-url="${encodeURIComponent(previewUrl)}" data-filepath="${encodeURIComponent(filepath)}" data-filename="${encodeURIComponent(filename)}">
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
                    ${previewHtml}
                    ${actionsHtml}
                `;
                filesContainer.appendChild(fileCard);
            }
        });
    }
    
    // Add to conversation history with files info for persistence
    const messageEntry = { role: 'assistant', content: content };
    if (allFiles.length > 0) {
        // Store file metadata for restoration (use URLs, not base64 to save space)
        messageEntry.files = allFiles.map(f => ({
            filename: f.filename || f.name,
            filepath: f.filepath || f.path,
            fileType: f.fileType || 'text',
            fileSize: f.fileSize,
            downloadUrl: f.downloadUrl || `${API_BASE_URL}/download?filepath=${encodeURIComponent(f.filepath || f.path)}`,
            previewUrl: f.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(f.filepath || f.path)}`,
            created: f.created !== false
        }));
    }
    conversationHistory.push(messageEntry);
    
    // Save chat after AI response immediately
    console.log('handleFinalAnswer: Saving chat with', conversationHistory.length, 'messages');
    saveCurrentChat();
    console.log('handleFinalAnswer: Chat saved');
}

function handleDone() {
    console.log('handleDone called, finalAnswerReceived:', finalAnswerReceived);
    setMessageStatus('✓ Complete', 'done');
    
    // If final_answer was never received but we have generated files, save them
    if (!finalAnswerReceived && generatedFiles.length > 0) {
        console.log('handleDone: Final answer not received, but have files. Creating fallback response.');
        const fallbackContent = 'Task completed. Files have been generated.';
        const messageEntry = { 
            role: 'assistant', 
            content: fallbackContent,
            files: generatedFiles.map(f => ({
                filename: f.filename || f.name,
                filepath: f.filepath || f.path,
                fileType: f.fileType || 'text',
                fileSize: f.fileSize,
                downloadUrl: f.downloadUrl || `${API_BASE_URL}/download?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                previewUrl: f.previewUrl || `${API_BASE_URL}/preview?filepath=${encodeURIComponent(f.filepath || f.path)}`,
                created: f.created !== false
            }))
        };
        conversationHistory.push(messageEntry);
        
        // Update the UI to show a final answer
        const answerSection = currentMessageElement?.querySelector('.answer-section');
        const finalAnswer = currentMessageElement?.querySelector('.final-answer');
        if (answerSection && finalAnswer) {
            answerSection.classList.remove('hidden');
            finalAnswer.textContent = fallbackContent;
        }
        
        saveCurrentChat();
        console.log('handleDone: Fallback response saved');
    }
    
    // Reset for next generation
    finalAnswerReceived = false;
    
    // Collapse plan and tasks after completion
    setTimeout(() => {
        const planSection = currentMessageElement?.querySelector('.plan-section');
        const tasksSection = currentMessageElement?.querySelector('.tasks-section');
        if (planSection) planSection.classList.add('collapsed');
        if (tasksSection) tasksSection.classList.add('collapsed');
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
    // Prevent any default navigation
    event?.preventDefault?.();
    event?.stopPropagation?.();
    
    try {
        console.log('Downloading file:', filepath, filename);
        const response = await fetch(`${API_BASE_URL}/download?filepath=${encodeURIComponent(filepath)}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Download failed: ${response.status} - ${errorText}`);
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename || 'download';
        document.body.appendChild(a);
        a.click();
        
        // Clean up after a delay to ensure download starts
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
            if (a.parentNode) {
                document.body.removeChild(a);
            }
        }, 100);
    } catch (error) {
        console.error('Download error:', error);
        alert('Failed to download file: ' + error.message);
    }
    
    return false;
}

// Start new chat
function startNewChat(event) {
    console.log('startNewChat called, event:', event?.type, 'isGenerating:', isGenerating);
    console.trace('startNewChat stack trace');
    
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    // Don't start new chat while generating
    if (isGenerating) {
        console.log('Cannot start new chat while generating');
        return false;
    }
    
    console.log('startNewChat: Clearing chat container');
    chatContainer.innerHTML = '';
    conversationHistory = [];
    currentMessageElement = null;
    currentChatId = null;
    promptInput.focus();
    renderChatHistory();
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
    if (e.key === 'Escape') {
        if (!mcpModal.classList.contains('hidden')) {
            closeMcpModal();
        }
        if (!projectModal.classList.contains('hidden')) {
            closeProjectModal();
        }
        if (!renameChatModal.classList.contains('hidden')) {
            closeRenameChatModal();
        }
    }
});

// Load MCP servers on page load
window.addEventListener('load', () => {
    loadMcpServers();
});

// ==========================================
// End MCP Server Management
// ==========================================

// ==========================================
// Sidebar & Project Event Listeners
// ==========================================

// Sidebar toggle
toggleSidebarBtn.addEventListener('click', toggleSidebar);
showSidebarBtn.addEventListener('click', showSidebar);

// Project management
newProjectBtn.addEventListener('click', () => openProjectModal());
editProjectBtn.addEventListener('click', () => openProjectModal(currentProjectId));
deleteProjectBtn.addEventListener('click', deleteProject);
projectSelect.addEventListener('change', (e) => {
    // Don't change project while generating
    if (isGenerating) {
        // Revert selection
        projectSelect.value = currentProjectId;
        alert('Cannot change project while generating');
        return;
    }
    currentProjectId = e.target.value;
    startNewChat();
    renderChatHistory();
});

// Project modal
projectModalClose.addEventListener('click', closeProjectModal);
projectModal.querySelector('.modal-overlay').addEventListener('click', closeProjectModal);
projectCancelBtn.addEventListener('click', closeProjectModal);
projectSaveBtn.addEventListener('click', saveProject);

// Rename chat modal
renameChatModalClose.addEventListener('click', closeRenameChatModal);
renameChatModal.querySelector('.modal-overlay').addEventListener('click', closeRenameChatModal);
renameChatCancelBtn.addEventListener('click', closeRenameChatModal);
renameChatSaveBtn.addEventListener('click', saveRenamedChat);

// Clear history
clearHistoryBtn.addEventListener('click', clearAllHistory);

// Mode selector change handler
if (modeSelect) {
    modeSelect.addEventListener('change', (e) => {
        const mode = e.target.value;
        updateModeHint(mode);
    });
}

// Update mode hint text
function updateModeHint(mode) {
    if (!modeHint) return;
    const hints = modeHint.querySelectorAll('span');
    hints.forEach(hint => hint.style.display = 'none');
    const activeHint = modeHint.querySelector(`.hint-${mode}`);
    if (activeHint) activeHint.style.display = 'inline';
}

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

// Focus input on load and initialize storage
window.addEventListener('load', () => {
    initStorage();
    promptInput.focus();
});

// Event delegation for download and preview buttons
document.addEventListener('click', (e) => {
    const target = e.target.closest('[data-action]');
    if (!target) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const action = target.dataset.action;
    
    if (action === 'download') {
        const filepath = decodeURIComponent(target.dataset.filepath || '');
        const filename = decodeURIComponent(target.dataset.filename || '');
        if (filepath && filename) {
            downloadFile(filepath, filename);
        }
    } else if (action === 'preview') {
        const url = decodeURIComponent(target.dataset.url || '');
        const filepath = decodeURIComponent(target.dataset.filepath || '');
        const filename = decodeURIComponent(target.dataset.filename || 'image');
        if (url) {
            openPreview(url, filename, filepath);
        }
    }
    
    return false;
});
