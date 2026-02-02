'use client';
// --- components/advisor.tsx ---
import { useState, useMemo, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Quiz {
  question: string;
  options: string[];
}

// Animated thinking messages - defined outside component to prevent re-creation
const THINKING_MESSAGES = [
  '🔍 Analyzing your question...',
  '📋 Assessment in process...',
  '🌾 Consulting crop experts...',
  '🐄 Checking livestock knowledge base...',  
  '📚 Retrieving breed database...',        
  '🧠 Processing recommendations...',
  '✨ Preparing your advice..',
];

export default function Advisor() {
  // Generate thread_id on client side only (after mount) to avoid hydration mismatch
  const [threadId, setThreadId] = useState('');
  
  const [input, setInput] = useState('');
  const [chat, setChat] = useState<Message[]>([
    { role: 'assistant', content: "Hello! I'm your AI agricultural assistant. How can I help you today?" }
  ]);
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [selectedOption, setSelectedOption] = useState<string>('');
  const [customAnswer, setCustomAnswer] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingMessage, setThinkingMessage] = useState('');

  // Generate unique thread_id on client side after component mounts
  useEffect(() => {
    if (!threadId) {
      setThreadId(`thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    }
  }, [threadId]);

  // Cycle through thinking messages
  useEffect(() => {
    if (!isThinking) {
      setThinkingMessage('');
      return;
    }

    let messageIndex = 0;
    setThinkingMessage(THINKING_MESSAGES[0]);

    const interval = setInterval(() => {
      messageIndex = (messageIndex + 1) % THINKING_MESSAGES.length;
      setThinkingMessage(THINKING_MESSAGES[messageIndex]);
    }, 1500); // Change message every 1.5 seconds

    return () => clearInterval(interval);
  }, [isThinking]); // Only depend on isThinking

  async function sendMessage(val: string, options?: { showUserBubble?: boolean }) {
    // Wait for threadId to be generated
    if (!threadId) {
      console.log('Waiting for thread ID to be generated...');
      return;
    }

    const showUserBubble = options?.showUserBubble ?? true;

    // Add user message to chat immediately (for free-text input)
    if (showUserBubble) {
      const userMessage: Message = { role: 'user', content: val };
      setChat(prev => [...prev, userMessage]);
    }
    
    // Clear input field and show thinking indicator
    setInput('');
    setIsThinking(true);
    // Hide quiz form immediately when submitting
    setQuiz(null);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: val, thread_id: threadId })
      });
      const data = await res.json();

      // Clear form fields after API response
      setSelectedOption('');
      setCustomAnswer('');

      if (data.status === 'requires_input') {
        // Show quiz form (question is included in the quiz form UI)
        setQuiz(data.ui); // This triggers the quiz form to show up
      } else if (data.status === 'complete') {
        // Format diagnosis and recommendations
        let responseContent = '';
        
        if (data.diagnosis) {
          // Clean up any remaining markdown artifacts
          responseContent = data.diagnosis
            .replace(/\*\*/g, '')  // Remove bold markdown
            .replace(/##\s+/g, '')  // Remove headers
            .replace(/\*/g, '')     // Remove asterisks
            .trim();
        }
        
        // Only add recommendations section if they exist and aren't already in diagnosis
        if (data.recommendations && data.recommendations.length > 0) {
          // Check if recommendations are already embedded in the diagnosis
          const hasEmbeddedRecs = data.recommendations.some((rec: string) => 
            responseContent.toLowerCase().includes(rec.toLowerCase().substring(0, 30))
          );
          
          if (!hasEmbeddedRecs) {
            responseContent += '\n\n📌 Quick Tips:\n';
            data.recommendations.slice(0, 3).forEach((rec: string, idx: number) => {
              const cleanRec = rec.replace(/\*\*/g, '').replace(/\*/g, '').trim();
              responseContent += `\n• ${cleanRec}`;
            });
          }
        }
        
        if (!responseContent) {
          responseContent = 'Assessment complete. How else can I help you?';
        }
        
        // Add final advice to chat
        const aiMessage: Message = { role: 'assistant', content: responseContent };
        setChat(prev => [...prev, aiMessage]);
      } else if (data.status === 'error') {
        // Handle error
        const errorMessage: Message = { 
          role: 'assistant', 
          content: `Sorry, an error occurred: ${data.message || 'Please try again.'}` 
        };
        setChat(prev => [...prev, errorMessage]);
      } else {
        // Fallback for unexpected status
        const aiMessage: Message = { role: 'assistant', content: 'Thank you for using the agricultural assistant!' };
        setChat(prev => [...prev, aiMessage]);
      }
    } finally {
      setIsThinking(false);
    }
  }

  function handleSubmitQuiz() {
    // Prioritize custom text if provided, otherwise use selected radio option
    const answer = customAnswer.trim() || selectedOption;
    if (!answer || !quiz) return;

    const currentQuiz = quiz;

    // Show question + submitted answer as a single assistant bubble
    setChat(prev => [
      ...prev,
      {
        role: 'assistant',
        content: `${currentQuiz.question}\n\nAnswer submitted: ${answer}`,
      },
    ]);

    // Clear form state immediately
    setSelectedOption('');
    setCustomAnswer('');

    // Send to backend without adding a separate user bubble
    // sendMessage will handle clearing quiz state and show thinking indicator
    sendMessage(answer, { showUserBubble: false });
  }

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Chat History Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chat.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-100'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ))}

        {/* Thinking Indicator with Animated Messages */}
        {isThinking && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg px-4 py-3 bg-gradient-to-r from-gray-800 to-gray-750 text-gray-100 shadow-lg">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <svg
                    className="animate-spin h-5 w-5 text-blue-400"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                </div>
                <span className="text-sm font-medium text-gray-200 animate-pulse">
                  {thinkingMessage || 'Processing...'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Quiz Form - appears inline after AI question */}
        {quiz && !isThinking && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg px-4 py-4 bg-gray-800 text-gray-100">
              <div className="space-y-4">
                {/* Question text */}
                <p className="text-sm text-gray-300 mb-2">{quiz.question}</p>
                
                {/* Radio buttons */}
                <div className="space-y-3">
                  {quiz.options.map((opt: string) => (
                    <label
                      key={opt}
                      className="flex items-center space-x-3 cursor-pointer hover:bg-gray-700 rounded p-2 -m-2"
                    >
                      <input
                        type="radio"
                        name="quiz-option"
                        value={opt}
                        checked={selectedOption === opt}
                        onChange={(e) => {
                          setSelectedOption(e.target.value);
                          setCustomAnswer(''); // Clear custom answer when radio is selected
                        }}
                        className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500 focus:ring-2"
                      />
                      <span className="text-sm">{opt}</span>
                    </label>
                  ))}
                </div>

                {/* Custom text input */}
                <div className="pt-2 border-t border-gray-700">
                  <label className="block text-sm text-gray-400 mb-2">
                    Or write your own answer...
                  </label>
                  <input
                    type="text"
                    value={customAnswer}
                    onChange={(e) => {
                      setCustomAnswer(e.target.value);
                      setSelectedOption(''); // Clear radio selection when typing
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && (customAnswer.trim() || selectedOption)) {
                        handleSubmitQuiz();
                      }
                    }}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Type your answer..."
                  />
                </div>

                {/* Submit button */}
                <button
                  onClick={handleSubmitQuiz}
                  disabled={!selectedOption && !customAnswer.trim()}
                  className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed rounded transition-colors text-sm font-medium"
                >
                  Submit
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area - only shown when no active quiz and not thinking */}
      {!quiz && !isThinking && (
        <div className="border-t border-gray-700 p-4">
          <div className="flex items-center space-x-2 max-w-2xl mx-auto">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && input.trim()) {
                  sendMessage(input);
                }
              }}
              className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Type your message..."
            />
            <button
              onClick={() => input.trim() && sendMessage(input)}
              disabled={!input.trim()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed rounded-lg transition-colors"
              aria-label="Send message"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
