# ===============================
# Base image
# ===============================
FROM python:3.10-slim

# ===============================
# Working directory
# ===============================
WORKDIR /app

# ===============================
# Install dependencies
# ===============================
COPY requirements_win.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_win.txt

# Install frameworks for dashboard
RUN pip install streamlit dash plotly pandas requests flask

# ===============================
# Copy application
# ===============================
COPY . /app/

# ===============================
# Expose ports
# ===============================
EXPOSE 8000
EXPOSE 8050
EXPOSE 8501

# ===============================
# Start API + Dash + Streamlit
# ===============================
CMD sh -c "\
python /app/serving/app.py & \
python /app/dashboard/dashboard/authority_dashboard.py & \
streamlit run /app/dashboard/dashboard/citizen_dashboard.py --server.port 8501 --server.enableCORS false \
"
