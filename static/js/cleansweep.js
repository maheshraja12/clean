(() => {
  "use strict";
  const fileInput = document.getElementById("file");
  const dropzone = document.getElementById("dropzone");
  const selected = document.getElementById("file-selected");
  const fileName = document.getElementById("file-name");
  const fileInfo = document.getElementById("file-info");
  const dropTitle = document.getElementById("drop-title");
  const removeButton = document.getElementById("remove-file");
  const form = document.getElementById("clean-form");
  const submitButton = document.getElementById("submit-button");
  const buttonLabel = document.getElementById("button-label");
  const missingAction = document.getElementById("missing_action");
  const missingDescription = document.getElementById("missing-description");
  const toast = document.getElementById("toast");
  const MAX_BYTES = 16 * 1024 * 1024;
  let toastTimer;

  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
  }

  function prettyBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function setFile(file) {
    if (!file) return;
    if (!/\.csv$/i.test(file.name)) {
      showToast("Please choose a CSV file (.csv).");
      if (fileInput) fileInput.value = "";
      return;
    }
    if (file.size > MAX_BYTES) {
      showToast("This file is larger than 16 MB. Choose a smaller CSV.");
      if (fileInput) fileInput.value = "";
      return;
    }
    if (fileInput && fileInput.files && fileInput.files[0] !== file) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
    }
    if (fileName) fileName.textContent = file.name;
    if (fileInfo) fileInfo.textContent = prettyBytes(file.size) + " · Ready to process";
    if (selected) selected.hidden = false;
    if (dropTitle) dropTitle.textContent = "File selected successfully";
    if (dropzone) dropzone.classList.add("has-file");
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (file) setFile(file);
    });
  }

  if (removeButton) {
    removeButton.addEventListener("click", () => {
      if (fileInput) fileInput.value = "";
      if (selected) selected.hidden = true;
      if (dropTitle) dropTitle.textContent = "Drop your CSV file here";
      if (dropzone) dropzone.classList.remove("has-file");
    });
  }

  if (dropzone) {
    ["dragenter", "dragover"].forEach(eventName => dropzone.addEventListener(eventName, event => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    }));
    ["dragleave", "drop"].forEach(eventName => dropzone.addEventListener(eventName, event => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    }));
    dropzone.addEventListener("drop", event => {
      const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) setFile(file);
    });
  }

  if (missingAction && missingDescription) {
    const descriptions = {
      ffill: "Fills gaps using the nearest available value, helping preserve rows.",
      mean: "Uses the average for numeric columns and the most common value for text.",
      drop: "Removes any row that still contains a missing value after duplicate removal."
    };
    missingAction.addEventListener("change", () => {
      missingDescription.textContent = descriptions[missingAction.value] || "Choose how missing values should be handled.";
    });
  }

  if (form) {
    form.addEventListener("submit", event => {
      const file = fileInput && fileInput.files && fileInput.files[0];
      if (!file) {
        event.preventDefault();
        showToast("Choose a CSV file before cleaning.");
        return;
      }
      if (file.size > MAX_BYTES) {
        event.preventDefault();
        showToast("This file is larger than 16 MB.");
        return;
      }
      if (submitButton) {
        submitButton.disabled = true;
        if (buttonLabel) buttonLabel.textContent = "Cleaning your dataset…";
      }
    });
  }
})();
