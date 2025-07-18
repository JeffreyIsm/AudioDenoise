import { useState } from "react";
import { Dropzone, FileMosaic } from "@files-ui/react";
import api from "../api/axios"; // Axios instance
import { getFullURL } from "../api/fullurl";

export default function Upload() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [resultURL, setResultURL] = useState(null);

  const updateFiles = (incomingFiles) => {
    setFiles(incomingFiles);
  };

  const removeFile = (id) => {
    setFiles(files.filter((file) => file.id !== id));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    const formData = new FormData();
    formData.append("file", files[0].file);

    setUploading(true);
    try {
      const res = await api.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResultURL({
        original: res.data.original_url,
        denoised: res.data.denoised_url,
      });
    } catch (err) {
      console.error("Upload failed", err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto flex flex-col items-center">
      <Dropzone
        onChange={updateFiles}
        value={files}
        label="🎤 Drop your noisy audio here or click to browse"
        maxFiles={1}
        accept=".wav"
        style={{
          border: "2px dashed #6366F1",
          borderRadius: "1rem",
          padding: "2rem",
          backgroundColor: "#F3F4F6",
          color: "#1F2937",
          fontSize: "1.1rem",
          fontWeight: "500",
          minHeight: "180px",
          cursor: "pointer",
          width: "100%",
        }}
      >
        {files.map((file) => (
          <FileMosaic key={file.id} {...file} onDelete={removeFile} info />
        ))}
      </Dropzone>

      {files.length > 0 && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="mt-6 w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-md transition-colors duration-200 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {uploading ? "Uploading..." : "Upload for Enhancement"}
        </button>
      )}

      {resultURL && (
        <div className="w-full mt-8 space-y-6">
          <div className="bg-green-50 rounded-lg p-4 shadow flex flex-col items-center">
            <p className="text-green-700 font-semibold mb-2">Original File:</p>
            <audio controls src={getFullURL(resultURL.original)} className="w-full rounded" />
          </div>

          <div className="bg-blue-50 rounded-lg p-4 shadow flex flex-col items-center">
            <p className="text-blue-700 font-semibold mb-2">Denoised File:</p>
            <audio controls src={getFullURL(resultURL.denoised)} className="w-full rounded" />
            <a
              href={getFullURL(resultURL.denoised)}
              download
              className="mt-3 inline-block px-4 py-2 bg-indigo-100 text-indigo-700 font-medium rounded hover:bg-indigo-200 transition-colors duration-200 text-sm shadow"
            >
              Download Denoised Audio
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
