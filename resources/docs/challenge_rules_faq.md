# EY Challenge 2026: Official Rules & FAQ

## Eligibility & Rules
- **Participants**: Open to university students and early career professionals with less than 5 years of experience.
- **Teams**: Teams can consist of up to 3 members.
- **Ownership**: Participants retain ownership of their IP but are encouraged to open-source winning solutions.
- **Prizes**:
  - **Winner**: $5,000
  - **1st Runner-up**: $3,000
  - **2nd Runner-up**: $2,000
  - **Snowflake Prize**: Invitation to Snowflake Summit 2026.

## Important Dates
- **Enrollment Opens**: Jan 20, 2026
- **Evaluation Starts**: March 14, 2026
- **Finalists Announced**: April 1, 2026
- **Winners Announced**: May 6, 2026

## Frequently Asked Questions (FAQ)
### What is the passing threshold?
An R² score of 0.4 is required to receive a certificate of completion.

### Can I use other tools?
Python is recommended. Snowflake is highly recommended but not mandatory. You can use any general notebook environment (AWS, Azure, Google Colab, etc.).

### Can I use external data?
Yes, participants can use any **free, publicly available** datasets.
- Examples: Landsat measurements, TerraClimate data, soil condition, air quality.
- **Constraint**: Data must be freely available to ensure reproducibility.

## Submission Details
- **Format**: CSV file matching the `submission_template.csv`.
- **Content**: Predictions for Total Alkalinity, Electrical Conductance, and Dissolved Reactive Phosphorus for the 200 target points.
- **Scoring**: Based on the average R² score across all three parameters.
