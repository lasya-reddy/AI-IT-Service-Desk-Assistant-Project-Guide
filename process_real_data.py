import pandas as pd
import re

url_sn = 'https://raw.githubusercontent.com/kavithacd/Auto-Ticket-Assignment/master/input_data.xlsx'
df_sn = pd.read_excel(url_sn)

# Real pattern mappings for real ServiceNow incident tickets
network_patterns = [
    r'\bvpn\b', r'\bwi-?fi\b', r'\bnetwork\b', r'\blan\b', r'\bdns\b', r'\bip\b',
    r'\bproxy\b', r'\bdisconnect', r'\bconnection\b', r'\btimeout\b', r'\binternet\b',
    r'\bgateway\b', r'\brouter\b', r'\bswitch\b', r'\bsubnet\b', r'\bping\b'
]

access_patterns = [
    r'\bpassword\b', r'\breset\b', r'\blogin\b', r'\block(ed)?\b', r'\baccess\b',
    r'\bpermission\b', r'\baccount\b', r'\bauth\b', r'\bmfa\b', r'\btoken\b',
    r'\bsso\b', r'\bcredentials?\b', r'\bunlock\b', r'\buser id\b', r'\brole\b'
]

hardware_patterns = [
    r'\blaptop\b', r'\bscreen\b', r'\bflicker\b', r'\bkeyboard\b', r'\bmouse\b',
    r'\bmonitor\b', r'\bprinter\b', r'\bprint\b', r'\bpaper jam\b', r'\bbattery\b',
    r'\bcharger\b', r'\bhardware\b', r'\bdesktop\b', r'\bpower\b', r'\bdock\b',
    r'\bheadset\b', r'\bmic\b', r'\bcamera\b', r'\bwebcam\b', r'\bdisk\b', r'\bhdd\b'
]

software_patterns = [
    r'\boutlook\b', r'\bexcel\b', r'\bteams\b', r'\bskype\b', r'\bsoftware\b',
    r'\binstall\b', r'\bcrash\b', r'\bfreeze\b', r'\berror\b', r'\bupdate\b',
    r'\bwindows\b', r'\bapp\b', r'\bapplication\b', r'\bbug\b', r'\btool\b',
    r'\boffice\b', r'\bword\b', r'\bpatch\b', r'\blicense\b', r'\bsync\b'
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'_x000D_', ' ', text)
    text = re.sub(r'received from:\s*\S+', ' ', text)
    text = re.sub(r'hello\s*team|hi\s*team|dear\s*team', ' ', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract text and classify
records = []
for _, row in df_sn.dropna(subset=['Short description']).iterrows():
    short_desc = clean_text(row['Short description'])
    full_desc = clean_text(str(row['Description']))
    text = short_desc
    if len(text) < 15 and len(full_desc) > len(text) and len(full_desc) < 200:
        text = f"{short_desc} - {full_desc}"
    
    if len(text) < 5 or text.lower().startswith('event:'):
        continue
        
    lower_text = text.lower()
    
    # Priority checks
    is_net = any(re.search(p, lower_text) for p in network_patterns)
    is_acc = any(re.search(p, lower_text) for p in access_patterns)
    is_hw = any(re.search(p, lower_text) for p in hardware_patterns)
    is_sw = any(re.search(p, lower_text) for p in software_patterns)
    
    # Exclusive / dominant categorisation
    if is_net and not is_acc and not is_hw:
        records.append((text, "Network"))
    elif is_acc and not is_net and not is_hw and not is_sw:
        records.append((text, "Access"))
    elif is_hw and not is_acc and not is_net:
        records.append((text, "Hardware"))
    elif is_sw and not is_net and not is_acc and not is_hw:
        records.append((text, "Software"))

df_filtered = pd.DataFrame(records, columns=['text', 'category'])
print(f"Extracted filtered real tickets: {len(df_filtered)}")
print(df_filtered['category'].value_counts())
print("\nSamples from each category:")
for cat in ['Network', 'Access', 'Hardware', 'Software']:
    print(f"\n--- {cat} ---")
    print(df_filtered[df_filtered['category'] == cat]['text'].head(3).tolist())
