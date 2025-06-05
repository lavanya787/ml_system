def run(self):
    st.title("🧠 Multi-Domain ML System")
    st.markdown("### Automatically detects and analyzes datasets from multiple domains")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section", [
        "Dataset Upload & Detection", "Data Analysis",
        "Model Training", "Intelligent Query",
        "Predictions", "System Logs"
    ])

    # Domain selection for manual override
    selected_domain = None
    if page == "Dataset Upload & Detection":
        selected_domain = st.sidebar.selectbox("Select domain (or Auto-Detect)", ['Auto-Detect'] + DOMAINS)

    # Page routing
    if page == "Dataset Upload & Detection":
        self.upload_and_detect_page(selected_domain)
    elif page == "Data Analysis":
        self.data_analysis_page()
    elif page == "Model Training":
        self.model_training_page()
    elif page == "Intelligent Query":
        self.intelligent_query_page()
    elif page == "Predictions":
        self.predictions_page()
    elif page == "System Logs":
        self.system_logs_page()

def upload_and_detect_page(self, selected_domain):
    st.subheader("📤 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format only)",
        type=["csv", "xls", "xlsx", "txt", "json", "pdf", "docx", "doc"]
    )

    if uploaded_file is not None:
        try:
            # For simplicity, try reading as CSV; handle other types as needed
            data = pd.read_csv(uploaded_file)
            self.logger.log_info(f"Dataset loaded successfully! Shape: {data.shape}")
            st.success(f"✅ Dataset loaded! Shape: {data.shape}")

            # Domain detection or manual selection
            if selected_domain == "Auto-Detect" or selected_domain is None:
                detection_result = self.detector.detect_domain(data)
                detected_domain = detection_result.get("domain")
                confidence = detection_result.get("confidence")
                st.write(f"Detected Domain: {detected_domain} (Confidence: {confidence:.2f})")
            else:
                detected_domain = selected_domain
                st.write(f"Manually selected domain: {detected_domain}")

            # Process domain-specific data
            if detected_domain:
                processed_data = self.processor.process_domain_data(data, detected_domain)
                st.session_state['processed_data'] = processed_data
                st.dataframe(processed_data.head())
            else:
                st.warning("No domain detected or selected. Processing skipped.")

        except Exception as e:
            self.logger.log_error(f"❌ Dataset upload failed: {str(e)}")
            st.error(f"❌ Error: {e}")
