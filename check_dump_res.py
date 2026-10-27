import pandas as pd

url_dump = 'https://raw.githubusercontent.com/pourabkarchaudhuri/itsm-ticket-classification/master/datasets/GMSCRFDump.csv'
df_dump = pd.read_csv(url_dump, encoding='latin1')

print("Columns in GMSCRFDump:", df_dump.columns.tolist())
df_sample = df_dump[['Title', 'Description', 'Service Category', 'Resolution Comments']].dropna().head(10)
for idx, row in df_sample.iterrows():
    print(f"Title: {row['Title']}")
    print(f"Cat: {row['Service Category']}")
    print(f"Res: {str(row['Resolution Comments'])[:120]}")
    print("-" * 30)
