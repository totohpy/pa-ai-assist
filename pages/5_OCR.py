import React, { useState } from 'react';

// Helper component สำหรับ Slider
const SliderInput = ({ label, min, max, step, value, onChange, unit = "" }) => (
  <div className="space-y-2">
    <label className="flex justify-between text-sm font-medium text-gray-600">
      <span>{label}</span>
      <span className="font-bold text-gray-800">{value}{unit}</span>
    </label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
    />
  </div>
);

// Helper component สำหรับ Spinner
const LoadingSpinner = () => (
  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

// Helper component สำหรับ Expander Container
const Expander = ({ title, children, isOpen, onToggle }) => (
  <div className="border border-gray-200 rounded-lg overflow-hidden mt-4">
    <button
      onClick={onToggle}
      className="w-full flex justify-between items-center p-4 bg-gray-50 hover:bg-gray-100 transition text-left focus:outline-none"
    >
      <span className="font-medium text-gray-700">{title}</span>
      <svg
        className={`w-5 h-5 text-gray-500 transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
    {isOpen && (
      <div className="p-4 bg-white space-y-6 border-t border-gray-200 animate-fade-in-down">
        {children}
      </div>
    )}
  </div>
);

// รับ 'apiKey' เป็น prop
export default function App({ apiKey }) {
  // State สำหรับเก็บค่าจาก form
  const [model, setModel] = useState('typhoon-ocr'); 
  const [taskType, setTaskType] = useState('v1.5'); 
  const [maxTokens, setMaxTokens] = useState(16000);
  const [temperature, setTemperature] = useState(0.1);
  const [topP, setTopP] = useState(0.6);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [pages, setPages] = useState('');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');

  // State สำหรับ UI
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // State สำหรับผลลัพธ์และสถานะ
  const [extractedText, setExtractedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [copySuccess, setCopySuccess] = useState('');

  // จัดการการเปลี่ยนแปลงไฟล์
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    } else {
      setFile(null);
      setFileName('No file chosen');
    }
  };

  // คัดลอกข้อความไปยัง clipboard
  const copyToClipboard = () => {
    const textArea = document.createElement('textarea');
    textArea.value = extractedText;
    textArea.style.position = 'fixed'; 
    textArea.style.opacity = 0;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      setCopySuccess('Copied to clipboard!');
    } catch (err) {
      setCopySuccess('Failed to copy!');
    }
    document.body.removeChild(textArea);
    setTimeout(() => setCopySuccess(''), 2000);
  };

  // ส่ง request ไปยัง OCR API
  const handleOcr = async () => {
    if (!file) {
      setError('Please upload a file first.');
      return;
    }
    if (!apiKey) {
      setError('API key is missing.');
      return;
    }

    setIsLoading(true);
    setError('');
    setExtractedText('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('task_type', taskType);
    formData.append('max_tokens', String(maxTokens));
    formData.append('temperature', String(temperature));
    formData.append('top_p', String(topP));
    formData.append('repetition_penalty', String(repetitionPenalty));
    if (pages.trim()) {
      formData.append('pages', pages.trim());
    }

    try {
      const response = await fetch("https://api.opentyphoon.ai/v1/ocr", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API Error ${response.status}: ${errText}`);
      }

      const result = await response.json();
      const extractedTexts = [];

      if (result.results && Array.isArray(result.results)) {
        for (const pageResult of result.results) {
          if (pageResult.success && pageResult.message?.choices?.[0]?.message?.content) {
            const content = pageResult.message.choices[0].message.content;
            try {
              const parsedContent = JSON.parse(content);
              const text = parsedContent.natural_text || JSON.stringify(parsedContent);
              extractedTexts.push(text);
            } catch (e) {
              extractedTexts.push(content);
            }
          } else if (!pageResult.success) {
            console.error(`Error processing ${pageResult.filename || 'unknown'}: ${pageResult.error || 'Unknown error'}`);
            extractedTexts.push(`[Error processing page: ${pageResult.error || 'Unknown error'}]`);
          }
        }
      } else {
        setError("Invalid response structure from API.");
      }

      setExtractedText(extractedTexts.join('\n\n---\n\n')); 
    } catch (err) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 sm:p-8 font-inter">
      <div className="max-w-7xl mx-auto bg-white p-6 sm:p-8 shadow-lg rounded-xl">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Typhoon OCR</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* --- คอลัมน์ซ้าย: Controls --- */}
          <div className="flex flex-col space-y-6">
            
            {/* File Upload (ย้ายขึ้นมาด้านบน) */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-2">
                Upload Image or PDF
              </label>
              <label className="w-full flex items-center justify-center p-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition">
                <svg className="w-6 h-6 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                <span className="text-gray-600">{fileName}</span>
                <input
                  type="file"
                  className="hidden"
                  onChange={handleFileChange}
                  accept="image/png, image/jpeg, image/webp, application/pdf"
                />
              </label>
            </div>

            {/* Pages (ย้ายมาต่อจาก Upload) */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-2">
                Pages (optional)
              </label>
              <input
                type="text"
                value={pages}
                onChange={(e) => setPages(e.target.value)}
                placeholder="e.g., [1, 2] or 1-3"
                className="w-full p-3 border border-gray-300 rounded-lg"
              />
            </div>
            
            {/* Start Button & Feedback */}
            <button
              onClick={handleOcr}
              disabled={isLoading}
              className="w-full flex items-center justify-center bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 transition shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? <LoadingSpinner /> : null}
              {isLoading ? 'Processing...' : 'Start OCR'}
            </button>
            
            {error && (
              <p className="text-red-500 text-sm text-center">{error}</p>
            )}

            {/* Advanced Settings Expander (ย้ายมาล่างสุด) */}
            <Expander 
              title="Advanced Settings" 
              isOpen={isSettingsOpen} 
              onToggle={() => setIsSettingsOpen(!isSettingsOpen)}
            >
              <SliderInput
                label="Max Tokens"
                min={1000} max={16000} step={100}
                value={maxTokens} onChange={setMaxTokens}
              />
              <SliderInput
                label="Temperature"
                min={0} max={1} step={0.1}
                value={temperature} onChange={setTemperature}
              />
              <SliderInput
                label="Top P"
                min={0} max={1} step={0.1}
                value={topP} onChange={setTopP}
              />
              <SliderInput
                label="Repetition Penalty"
                min={1} max={2} step={0.1}
                value={repetitionPenalty} onChange={setRepetitionPenalty}
              />
            </Expander>

          </div>
          
          {/* --- คอลัมน์ขวา: Output --- */}
          <div className="flex flex-col">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-gray-600">
                Extracted Text
              </label>
              <div className="flex items-center">
                {copySuccess && <span className="text-green-500 text-sm mr-2">{copySuccess}</span>}
                <button
                  onClick={copyToClipboard}
                  disabled={!extractedText}
                  className="bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded-lg hover:bg-gray-300 transition text-sm disabled:opacity-50"
                >
                  Copy
                </button>
              </div>
            </div>
            <textarea
              readOnly
              value={extractedText}
              placeholder="Your extracted text will appear here..."
              className="w-full h-full flex-grow p-4 border border-gray-300 rounded-lg bg-gray-50 min-h-[500px]"
            />
          </div>
          
        </div>
      </div>
    </div>
  );
}
