---
 
## 📄 Document Classifier using Local Transformers
 
This project is a powerful document classification tool built in Python that uses **local Hugging Face transformer models**. It supports multiple file types including PDFs, Word documents, scanned images, and email files — classifying them into predefined categories such as invoices, resumes, legal documents, and more.
 
---
---
 
### 📦 Supported File Types
 
| File Type | Handled By              |
| --------- | ----------------------- |
| PDF       | `pdfplumber`            |
| DOCX      | `python-docx`           |
| Images    | `pytesseract`, `opencv` |
| EML       | `pyzmail36`             |
 
---

---
 
### 🔍 Sample Output
 
```
📄 Classifying: resume.docx
🔎 Predicted Category: employment resume
Top 3 Predictions: ['employment resume', 'curriculum vitae', 'official letter']
```
 
---
