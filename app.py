import torch
import gc
import pandas as pd
import os
import glob
import time
import platform
import psutil

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

clear_gpu_memory()

file_path = r"C:\Users\HuyNguyen\Documents\CODE\ff\data.csv"

# Create separate output directories for each table
base_output_dir = r"C:\Users\HuyNguyen\Documents\CODE\ff\cleaned_files"
os.makedirs(base_output_dir, exist_ok=True)

# Define the common output directory structure
output_dir = os.path.join(base_output_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True)

# Create more refined directory structure with primary and lookup tables
directories = {
    # Core tables
    "customers": os.path.join(output_dir, "customers"),  # Primary customer table
    "accounts": os.path.join(output_dir, "accounts"),  # Account status over time
    
    # Lookup tables
    "countries": os.path.join(output_dir, "lookup_tables", "countries"),
    "provinces": os.path.join(output_dir, "lookup_tables", "provinces"),
    "employee_types": os.path.join(output_dir, "lookup_tables", "employee_types"),
    "channels": os.path.join(output_dir, "lookup_tables", "channels"),
    "segments": os.path.join(output_dir, "lookup_tables", "segments"),
    "relationship_types": os.path.join(output_dir, "lookup_tables", "relationship_types"),
    
    # Association tables
    "customer_locations": os.path.join(output_dir, "association_tables", "customer_locations"),
    "customer_employees": os.path.join(output_dir, "association_tables", "customer_employees"),
    "customer_relationships": os.path.join(output_dir, "association_tables", "customer_relationships"),
    
    # Product categories
    "banking_products": os.path.join(output_dir, "product_categories", "banking_products"),
    "investment_products": os.path.join(output_dir, "product_categories", "investment_products"),
    "loan_products": os.path.join(output_dir, "product_categories", "loan_products"),
    "payment_services": os.path.join(output_dir, "product_categories", "payment_services")
}

# Create all directories
for directory in directories.values():
    os.makedirs(directory, exist_ok=True)

chunk_size = 500000

# Column mappings for renaming
rename_columns = {
    "fecha_dato": "date_record",
    "ncodpers": "customer_code",
    "ind_empleado": "employee_index",
    "pais_residencia": "residence_country",
    "sexo": "gender",
    "age": "age",
    "fecha_alta": "first_join_date",
    "ind_nuevo": "new_customer_index",
    "antiguedad": "seniority",
    "indrel": "customer_type",
    "ult_fec_cli_1t": "last_primary_customer_date",
    "indrel_1mes": "customer_status",
    "tiprel_1mes": "customer_relationship",
    "indresi": "residence_flag",
    "indext": "foreigner_flag",
    "conyuemp": "spouse_index",
    "canal_entrada": "channel_code",
    "indfall": "deceased_flag",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "customer_activity",
    "renta": "income",
    "segmento": "segment_code",
    "ind_ahor_fin_ult1": "savings_account",
    "ind_aval_fin_ult1": "guarantees",
    "ind_cco_fin_ult1": "current_account",
    "ind_cder_fin_ult1": "derivatives_account",
    "ind_cno_fin_ult1": "payroll_account",
    "ind_ctju_fin_ult1": "junior_account",
    "ind_ctma_fin_ult1": "more_particular_account",
    "ind_ctop_fin_ult1": "particular_account",
    "ind_ctpp_fin_ult1": "private_account",
    "ind_deco_fin_ult1": "short_term_deposits",
    "ind_deme_fin_ult1": "medium_term_deposits",
    "ind_dela_fin_ult1": "long_term_deposits",
    "ind_ecue_fin_ult1": "e_account",
    "ind_fond_fin_ult1": "investment_funds",
    "ind_hip_fin_ult1": "mortgage",
    "ind_plan_fin_ult1": "pensions",
    "ind_pres_fin_ult1": "loans",
    "ind_reca_fin_ult1": "taxes_and_collections",
    "ind_tjcr_fin_ult1": "credit_card",
    "ind_valo_fin_ult1": "securities",
    "ind_viv_fin_ult1": "home_loan",
    "ind_nomina_ult1": "payroll",
    "ind_nom_pens_ult1": "pension_payroll",
    "ind_recibo_ult1": "direct_debit"
}

# Define column data types
column_types = {
    "ind_ahor_fin_ult1": int,
    "ind_aval_fin_ult1": int,
    "ind_cco_fin_ult1": int,
    "ind_cder_fin_ult1": int,
    "ind_cno_fin_ult1": int,
    "ind_ctju_fin_ult1": int,
    "ind_ctma_fin_ult1": int,
    "ind_ctop_fin_ult1": int,
    "ind_ctpp_fin_ult1": int,
    "ind_deco_fin_ult1": int,
    "ind_deme_fin_ult1": int,
    "ind_dela_fin_ult1": int,
    "ind_ecue_fin_ult1": int,
    "ind_fond_fin_ult1": int,
    "ind_hip_fin_ult1": int,
    "ind_plan_fin_ult1": int,
    "ind_pres_fin_ult1": int,
    "ind_reca_fin_ult1": int,
    "ind_tjcr_fin_ult1": int,
    "ind_valo_fin_ult1": int,
    "ind_viv_fin_ult1": int,
    "ind_nomina_ult1": int,
    "ind_nom_pens_ult1": int,
    "ind_recibo_ult1": int
}

# Define lookup tables and their columns
lookup_tables = {
    "countries": ["country_id", "country_code", "country_name"],
    "provinces": ["province_id", "province_code", "province_name"],
    "employee_types": ["employee_type_id", "employee_code", "employee_description"],
    "channels": ["channel_id", "channel_code", "channel_description"],
    "segments": ["segment_id", "segment_code", "segment_description"],
    "relationship_types": ["relationship_type_id", "relationship_code", "relationship_description"]
}

# Initialize lookup tables with predefined data
country_lookup = pd.DataFrame()
province_lookup = pd.DataFrame()
employee_type_lookup = pd.DataFrame({
    "employee_type_id": [1, 2, 3, 4, 5],
    "employee_code": ["A", "B", "F", "N", "S"],
    "employee_description": ["Active", "Ex-employee", "Branch Employee", "Not Employee", "Self-employed"]
})
channel_lookup = pd.DataFrame()
segment_lookup = pd.DataFrame({
    "segment_id": [1, 2, 3],
    "segment_code": ["01 - TOP", "02 - INDIVIDUAL", "03 - STUDENT"],
    "segment_description": ["Top Client", "Individual Client", "Student"]
})
relationship_lookup = pd.DataFrame({
    "relationship_type_id": [1, 2, 3, 4],
    "relationship_code": ["A", "I", "P", "R"],
    "relationship_description": ["Active", "Inactive", "Former", "Potential"]
})

# Define columns for each main table
customer_columns = [
    "customer_id", "customer_code", "first_join_date", "gender", "age", 
    "income", "segment_id", "deceased_flag"
]

account_columns = [
    "account_id", "customer_id", "date_record", "new_customer_index", 
    "seniority", "customer_type", "customer_status", "customer_activity"
]

customer_location_columns = [
    "location_id", "customer_id", "country_id", "province_id", "residence_flag", 
    "foreigner_flag", "address_type"
]

customer_employee_columns = [
    "customer_employee_id", "customer_id", "employee_type_id", "date_record"
]

customer_relationship_columns = [
    "relationship_id", "customer_id", "relationship_type_id", "spouse_index", 
    "channel_id", "date_record"
]

banking_product_columns = [
    "product_id", "customer_id", "date_record", "savings_account", "current_account", 
    "junior_account", "more_particular_account", "particular_account", 
    "private_account", "e_account"
]

investment_product_columns = [
    "product_id", "customer_id", "date_record", "short_term_deposits", 
    "medium_term_deposits", "long_term_deposits", "investment_funds", 
    "securities", "pensions"
]

loan_product_columns = [
    "product_id", "customer_id", "date_record", "mortgage", "loans", 
    "home_loan", "guarantees"
]

payment_service_columns = [
    "service_id", "customer_id", "date_record", "credit_card", "payroll_account", 
    "payroll", "pension_payroll", "direct_debit", "taxes_and_collections", 
    "derivatives_account"
]

# Create or clear processing log file

system_info = {
    "OS": platform.system(),
    "OS Version": platform.version(),
    "Machine Type": platform.machine(),
    "Processor": platform.processor(),
    "Memory (Total)": psutil.virtual_memory().total,
    "Memory (Available)": psutil.virtual_memory().available,
}

with open(os.path.join(base_output_dir, 'processing_log.txt'), 'w') as log_file:
    log_file.write("Processing started at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log_file.write("\n--- System Configuration ---\n")
    for key, value in system_info.items():
        log_file.write(f"{key}: {value}\n")

start_time = time.time()

# Create a temporary file to store income mean for later use
temp_income_file = os.path.join(base_output_dir, "temp_income_stats.csv")
income_stats_calculated = False

# Customer ID mapping dictionary to maintain ID consistency across chunks
customer_id_mapping = {}
country_id_mapping = {}
province_id_mapping = {}
channel_id_mapping = {}

# Function to get or create a mapping ID
def get_or_create_id(mapping_dict, key, prefix=''):
    if key in mapping_dict:
        return mapping_dict[key]
    else:
        new_id = len(mapping_dict) + 1
        mapping_dict[key] = new_id
        return new_id

# Process each chunk
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, dtype=str, encoding="utf-8")):
    chunk_start_time = time.time()
    
    # Rename columns
    chunk.rename(columns=rename_columns, inplace=True)

    # Change data types
    for column, dtype in column_types.items():
        if column in chunk.columns:  # Kiểm tra cột có tồn tại trong chunk không
            chunk[column] = chunk[column].astype(dtype)
    
    # Clean up employee_index
    employee_map = {
        'A': 'A', 
        'B': 'B', 
        'F': 'F', 
        'N': 'N', 
        'S': 'S'
    }
    chunk['employee_index'] = chunk['employee_index'].map(employee_map).fillna(chunk['employee_index'])
    
    # Clean up customer_relationship
    relationship_map = {
        'A': 'A',
        'I': 'I',
        'P': 'P',
        'R': 'R'
    }
    chunk['customer_relationship'] = chunk['customer_relationship'].map(relationship_map).fillna(chunk['customer_relationship'])
    
    # Clean up flags (residence_flag, foreigner_flag, deceased_flag)
    flag_map = {
        'N': 0,
        'S': 1
    }
    for flag_col in ['residence_flag', 'foreigner_flag', 'deceased_flag']:
        if flag_col in chunk.columns:
            chunk[flag_col] = chunk[flag_col].map(flag_map).fillna(0).astype(int)
    
    # Clean up gender
    gender_map = {
        'H': 'M',
        'V': 'F'
    }
    chunk['gender'] = chunk['gender'].map(gender_map).fillna(chunk['gender'])
    chunk['gender'] = chunk['gender'].fillna('U')
    
    # Convert numeric fields safely
    chunk['age'] = pd.to_numeric(chunk['age'], errors='coerce')
    chunk['seniority'] = pd.to_numeric(chunk['seniority'], errors='coerce').fillna(0).astype(int)
    chunk['income'] = pd.to_numeric(chunk['income'], errors='coerce')
    
    # Handle income: fill missing values with mean
    if not income_stats_calculated:
        # Calculate mean from non-missing values
        income_mean = chunk['income'].dropna().mean()
        # Save mean to temporary file for later chunks
        pd.DataFrame({'income_mean': [income_mean]}).to_csv(temp_income_file, index=False)
        income_stats_calculated = True
    else:
        # For subsequent chunks, use the mean from the first chunk
        income_stats = pd.read_csv(temp_income_file)
        if not income_stats.empty:
            income_mean = income_stats['income_mean'].iloc[0]
        else:
            # Fallback if file read fails
            income_mean = chunk['income'].dropna().mean()

    # Fill missing income values with the calculated mean
    chunk['income'] = chunk['income'].fillna(income_mean)
    
    # Clean up spouse_index and channel_code
    chunk['spouse_index'] = chunk['spouse_index'].fillna('N')
    chunk['channel_code'] = chunk['channel_code'].fillna('UNK').astype(str)
    
    # Clean up province_code and province_name
    chunk['province_code'] = chunk['province_code'].fillna('UNK')
    chunk['province_name'] = chunk['province_name'].fillna('Unknown')
    
    # Apply segment rules
    chunk['segment_code'] = chunk['segment_code'].fillna('')
    mask_young = chunk['age'] < 25
    mask_high_income = chunk['income'] > income_mean
    
    chunk.loc[mask_young & chunk['segment_code'].isin(['', 'nan']), 'segment_code'] = '03 - STUDENT'
    chunk.loc[mask_high_income & ~mask_young & chunk['segment_code'].isin(['', 'nan']), 'segment_code'] = '01 - TOP'
    chunk.loc[~mask_high_income & ~mask_young & chunk['segment_code'].isin(['', 'nan']), 'segment_code'] = '02 - INDIVIDUAL'
    
    # Translate existing Spanish segment names to English
    segment_translation = {
        '03 - UNIVERSITARIO': '03 - STUDENT',
        '01 - TOP': '01 - TOP',
        '02 - PARTICULARES': '02 - INDIVIDUAL'
    }
    chunk['segment_code'] = chunk['segment_code'].map(lambda x: segment_translation.get(x, x) if x and x != 'nan' else '02 - INDIVIDUAL')
    
    # Drop rows with too many missing values
    chunk_cleaned = chunk.dropna(thresh=len(chunk.columns) - 10).copy()
    
    # Handle binary indicator columns (convert to 0/1)
    binary_columns = [
        "new_customer_index", "customer_activity", "savings_account", "guarantees",
        "current_account", "derivatives_account", "payroll_account", "junior_account",
        "more_particular_account", "particular_account", "private_account", "short_term_deposits", "medium_term_deposits",
        "long_term_deposits", "e_account", "investment_funds", "mortgage",
        "pensions", "loans", "taxes_and_collections", "credit_card", "securities",
        "home_loan", "payroll", "pension_payroll", "direct_debit"
    ]
    for col in binary_columns:
        if col in chunk_cleaned.columns:
            chunk_cleaned[col] = pd.to_numeric(chunk_cleaned[col], errors='coerce').fillna(0).astype(int)
    
    # Generate consistent IDs for customers
    # Create or update customer ID mapping
    customer_ids = {}
    for idx, customer_code in enumerate(chunk_cleaned['customer_code'].unique()):
        if customer_code not in customer_id_mapping:
            customer_id_mapping[customer_code] = len(customer_id_mapping) + 1
        customer_ids[customer_code] = customer_id_mapping[customer_code]
    
    # Apply customer ID mapping
    chunk_cleaned['customer_id'] = chunk_cleaned['customer_code'].map(customer_ids)
    
    # Create lookup tables for countries, provinces, channels
    if len(country_lookup) == 0:
        unique_countries = chunk_cleaned['residence_country'].dropna().unique()
        country_lookup = pd.DataFrame({
            'country_id': range(1, len(unique_countries) + 1),
            'country_code': unique_countries,
            'country_name': unique_countries  # Using code as name initially
        })
        # Save country lookup table
        country_lookup.to_csv(os.path.join(directories['countries'], 'country_lookup.csv'), index=False)
    
    if len(province_lookup) == 0:
        provinces = chunk_cleaned[['province_code', 'province_name']].drop_duplicates()
        provinces = provinces.dropna(subset=['province_code'])
        province_lookup = pd.DataFrame({
            'province_id': range(1, len(provinces) + 1),
            'province_code': provinces['province_code'].values,
            'province_name': provinces['province_name'].values
        })
        # Save province lookup table
        province_lookup.to_csv(os.path.join(directories['provinces'], 'province_lookup.csv'), index=False)
    
    if len(channel_lookup) == 0:
        unique_channels = chunk_cleaned['channel_code'].dropna().unique()
        channel_lookup = pd.DataFrame({
            'channel_id': range(1, len(unique_channels) + 1),
            'channel_code': unique_channels,
            'channel_description': ['Channel ' + str(code) for code in unique_channels]
        })
        # Save channel lookup table
        channel_lookup.to_csv(os.path.join(directories['channels'], 'channel_lookup.csv'), index=False)
    
    # Save employee_type_lookup
    employee_type_lookup.to_csv(os.path.join(directories['employee_types'], 'employee_type_lookup.csv'), index=False)
    
    # Save segment_lookup
    segment_lookup.to_csv(os.path.join(directories['segments'], 'segment_lookup.csv'), index=False)
    
    # Save relationship_lookup
    relationship_lookup.to_csv(os.path.join(directories['relationship_types'], 'relationship_type_lookup.csv'), index=False)
    
    # Create mapping dictionaries from lookup tables
    country_mapping = dict(zip(country_lookup['country_code'], country_lookup['country_id']))
    province_mapping = dict(zip(province_lookup['province_code'], province_lookup['province_id']))
    employee_mapping = dict(zip(employee_type_lookup['employee_code'], employee_type_lookup['employee_type_id']))
    channel_mapping = dict(zip(channel_lookup['channel_code'], channel_lookup['channel_id']))
    segment_mapping = dict(zip(segment_lookup['segment_code'], segment_lookup['segment_id']))
    relationship_mapping = dict(zip(relationship_lookup['relationship_code'], relationship_lookup['relationship_type_id']))
    
    # Create main tables
    
    # Customers table
    customers = chunk_cleaned[['customer_id', 'customer_code', 'first_join_date', 'gender', 'age', 
                              'income', 'segment_code', 'deceased_flag']].drop_duplicates('customer_id').copy()
    
    # Map segment code to segment ID
    customers['segment_id'] = customers['segment_code'].map(segment_mapping)
    customers = customers[customer_columns]
    
    # Accounts table
    accounts = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        accounts = chunk_cleaned[['customer_id', 'date_record', 'new_customer_index', 
                                 'seniority', 'customer_type', 'customer_status', 'customer_activity']].copy()
        accounts['account_id'] = range(1, len(accounts) + 1)
        accounts = accounts[account_columns]
    
    # Customer locations table
    customer_locations = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        customer_locations = chunk_cleaned[['customer_id', 'residence_country', 'province_code', 
                                           'residence_flag', 'foreigner_flag', 'address_type']].drop_duplicates('customer_id').copy()
        # Map country and province codes to IDs
        customer_locations['country_id'] = customer_locations['residence_country'].map(country_mapping)
        customer_locations['province_id'] = customer_locations['province_code'].map(province_mapping)
        customer_locations['location_id'] = range(1, len(customer_locations) + 1)
        customer_locations = customer_locations[customer_location_columns]
    
    # Customer employees table
    customer_employees = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        customer_employees = chunk_cleaned[['customer_id', 'employee_index', 'date_record']].copy()
        customer_employees['employee_type_id'] = customer_employees['employee_index'].map(employee_mapping)
        customer_employees = customer_employees.dropna(subset=['employee_type_id'])
        customer_employees['customer_employee_id'] = range(1, len(customer_employees) + 1)
        customer_employees = customer_employees[customer_employee_columns]
    
    # Customer relationships table
    customer_relationships = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        customer_relationships = chunk_cleaned[['customer_id', 'customer_relationship', 'spouse_index', 
                                               'channel_code', 'date_record']].copy()
        customer_relationships['relationship_type_id'] = customer_relationships['customer_relationship'].map(relationship_mapping)
        customer_relationships['channel_id'] = customer_relationships['channel_code'].map(channel_mapping)
        customer_relationships['relationship_id'] = range(1, len(customer_relationships) + 1)
        customer_relationships = customer_relationships[customer_relationship_columns]
    
    # Banking products table
    banking_products = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        banking_products = chunk_cleaned[['customer_id', 'date_record', 'savings_account', 'current_account', 
                                         'junior_account', 'more_particular_account', 'particular_account', 
                                         'private_account', 'e_account']].copy()
        banking_products['product_id'] = range(1, len(banking_products) + 1)
        banking_products = banking_products[banking_product_columns]
    
    # Investment products table
    investment_products = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        investment_products = chunk_cleaned[['customer_id', 'date_record', 'short_term_deposits', 
                                            'medium_term_deposits', 'long_term_deposits', 'investment_funds', 
                                            'securities', 'pensions']].copy()
        investment_products['product_id'] = range(1, len(investment_products) + 1)
        investment_products = investment_products[investment_product_columns]
    
    # Loan products table
    loan_products = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        loan_products = chunk_cleaned[['customer_id', 'date_record', 'mortgage', 'loans', 
                                      'home_loan', 'guarantees']].copy()
        loan_products['product_id'] = range(1, len(loan_products) + 1)
        loan_products = loan_products[loan_product_columns]
    
    # Payment services table
    payment_services = pd.DataFrame()
    if len(chunk_cleaned) > 0:
        payment_services = chunk_cleaned[['customer_id', 'date_record', 'credit_card', 'payroll_account', 
                                         'payroll', 'pension_payroll', 'direct_debit', 'taxes_and_collections', 
                                         'derivatives_account']].copy()
        payment_services['service_id'] = range(1, len(payment_services) + 1)
        payment_services = payment_services[payment_service_columns]
    
    # Generate unique filenames for each chunk
    chunk_id = f"{i+1:03d}"
    
    # Save tables to CSV files
    customers.to_csv(os.path.join(directories['customers'], f'customers_{chunk_id}.csv'), index=False)
    
    if len(accounts) > 0:
        accounts.to_csv(os.path.join(directories['accounts'], f'accounts_{chunk_id}.csv'), index=False)
    
    if len(customer_locations) > 0:
        customer_locations.to_csv(os.path.join(directories['customer_locations'], f'customer_locations_{chunk_id}.csv'), index=False)
    
    if len(customer_employees) > 0:
        customer_employees.to_csv(os.path.join(directories['customer_employees'], f'customer_employees_{chunk_id}.csv'), index=False)
    
    if len(customer_relationships) > 0:
        customer_relationships.to_csv(os.path.join(directories['customer_relationships'], f'customer_relationships_{chunk_id}.csv'), index=False)
    
    if len(banking_products) > 0:
        banking_products.to_csv(os.path.join(directories['banking_products'], f'banking_products_{chunk_id}.csv'), index=False)
    
    if len(investment_products) > 0:
        investment_products.to_csv(os.path.join(directories['investment_products'], f'investment_products_{chunk_id}.csv'), index=False)
    
    if len(loan_products) > 0:
        loan_products.to_csv(os.path.join(directories['loan_products'], f'loan_products_{chunk_id}.csv'), index=False)
    
    if len(payment_services) > 0:
        payment_services.to_csv(os.path.join(directories['payment_services'], f'payment_services_{chunk_id}.csv'), index=False)
    
    # Release memory
    del chunk, chunk_cleaned, customers, accounts, customer_locations
    del customer_employees, customer_relationships, banking_products
    del investment_products, loan_products, payment_services
    clear_gpu_memory()
    
    # Log progress
    chunk_duration = time.time() - chunk_start_time
    with open(os.path.join(base_output_dir, 'processing_log.txt'), 'a') as log_file:
        log_file.write(f"Completed chunk {i+1} in {chunk_duration:.2f} seconds\n")
    print(f"Completed chunk {i+1} in {chunk_duration:.2f} seconds")

# Delete the temporary income stats file
try:
    os.remove(temp_income_file)
except:
    pass

# Consolidate all files into single tables
for table_name, directory in directories.items():
    file_pattern = os.path.join(directory, '*.csv')
    all_files = glob.glob(file_pattern)
    
    if all_files:
        # Initialize an empty list to store each dataframe
        all_dfs = []
        
        # Read each CSV file and append to list
        for file in all_files:
            if os.path.basename(file).startswith(f"{table_name}_"):
                df = pd.read_csv(file, low_memory=False)
                all_dfs.append(df)
        
        # Combine all dataframes
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            # Save combined file
            combined_df.to_csv(os.path.join(directory, f"{table_name}_full.csv"), index=False)
            
            # Log the consolidation
            with open(os.path.join(base_output_dir, 'processing_log.txt'), 'a') as log_file:
                log_file.write(f"Consolidated {len(all_dfs)} files for {table_name}\n")

# Calculate total processing time
total_duration = time.time() - start_time
with open(os.path.join(base_output_dir, 'processing_log.txt'), 'a') as log_file:
    log_file.write(f"Total processing time: {total_duration:.2f} seconds\n")
    log_file.write("Processing completed at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

print(f"Total processing time: {total_duration:.2f} seconds")
print("Processing completed!")

# Create metadata file with schema information
metadata = {
    "tables": {
        "customers": {"columns": customer_columns, "description": "Primary customer information"},
        "accounts": {"columns": account_columns, "description": "Customer account status over time"},
        "customer_locations": {"columns": customer_location_columns, "description": "Customer geographical information"},
        "customer_employees": {"columns": customer_employee_columns, "description": "Employment status of customers"},
        "customer_relationships": {"columns": customer_relationship_columns, "description": "Customer relationship types and channels"},
        "banking_products": {"columns": banking_product_columns, "description": "Banking products owned by customers"},
        "investment_products": {"columns": investment_product_columns, "description": "Investment products owned by customers"},
        "loan_products": {"columns": loan_product_columns, "description": "Loan products owned by customers"},
        "payment_services": {"columns": payment_service_columns, "description": "Payment services used by customers"}
    },
    "lookup_tables": {
        "countries": {"columns": lookup_tables["countries"], "description": "Country reference data"},
        "provinces": {"columns": lookup_tables["provinces"], "description": "Province reference data"},
        "employee_types": {"columns": lookup_tables["employee_types"], "description": "Employee type reference data"},
        "channels": {"columns": lookup_tables["channels"], "description": "Channel reference data"},
        "segments": {"columns": lookup_tables["segments"], "description": "Segment reference data"},
        "relationship_types": {"columns": lookup_tables["relationship_types"], "description": "Relationship type reference data"}
    },
    "processing_info": {
        "total_processing_time": f"{total_duration:.2f} seconds",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_size": chunk_size
    }
}

# Write metadata to JSON file
import json
with open(os.path.join(base_output_dir, 'schema_metadata.json'), 'w') as json_file:
    json.dump(metadata, json_file, indent=4)

print("Schema metadata created successfully!")

# Create a README file with database structure information
readme_content = f"""# Financial Services Database

## Overview
This database contains financial services customer data structured in a normalized format.
Data processing completed on {time.strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure
- `customers/` : Contains customer information files.
- `accounts/` : Stores customer account history.
- `lookup_tables/` : Stores reference tables such as countries, provinces, and segments.
- `association_tables/` : Stores relationships between different entities.
- `product_categories/` : Contains different financial product categories.

## Data Processing
The data was processed in chunks, cleaned, and structured into separate files.
Each chunk was processed and stored individually before being consolidated into full datasets.

## Files Generated
- `customers_full.csv` : Consolidated customer data.
- `accounts_full.csv` : Account data history.
- `banking_products_full.csv` : Banking-related products.
- `investment_products_full.csv` : Investment-related products.
- `loan_products_full.csv` : Loan-related data.
- `payment_services_full.csv` : Payment services data.
- `schema_metadata.json` : Contains metadata about the database structure.
- `processing_log.txt` : Log file tracking processing time and errors.

## Notes
- Missing income values were replaced with the mean income.
- Certain categorical values were mapped to predefined codes.
- Binary indicators were converted to `0/1` format.

## Contact
For any questions or concerns, please reach out to the data engineering team."""

# Save the README file
readme_path = os.path.join(base_output_dir, "README.txt")
with open(readme_path, "w", encoding="utf-8") as readme_file:
    readme_file.write(readme_content)

print("README file created successfully!")