"""
Synthetic Data Generator for E-Commerce / Subscription MarTech CDP
Generates:
- Raw Profiles (Customers with PII & demographics)
- Identity Touchpoints (Cookie, Device ID, Phone, CRM ID, Email) with realistic fragmented linkages
- Transactions / Purchases (Order history, revenue, margin)
- Web & App Events (Page views, product clicks, cart adds, pricing visits)
- Campaign Interactions (Paid Search, Social, Email, Display touches)
- Consent Preferences (Marketing opt-in, tracking consent, channel consent)
"""
import uuid
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

def generate_cdp_dataset(
    num_customers: int = 1500,
    start_date: datetime = datetime(2025, 1, 1),
    end_date: datetime = datetime(2026, 3, 1),
):
    print(f"Generating synthetic MarTech CDP dataset for {num_customers} true entities...")
    
    # 1. Generate True Entities
    entities = []
    channels = ["paid_search", "paid_social", "organic_search", "email_marketing", "direct", "referral", "display"]
    tiers = ["standard", "premium", "vip"]
    
    for i in range(num_customers):
        entity_id = f"ENT_{i+1:05d}"
        created_at = fake.date_time_between(start_date=start_date, end_date=end_date - timedelta(days=90))
        
        # Primary attributes
        first_name = fake.first_name()
        last_name = fake.last_name()
        primary_email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@{fake.free_email_domain()}"
        primary_phone = fake.phone_number()
        city = fake.city()
        country = "US"
        
        # Customer behavioral propensity traits
        churn_propensity = np.random.beta(2, 5) # most are active/moderate
        cltv_archetype = np.random.choice(["low", "medium", "high", "vip"], p=[0.50, 0.30, 0.15, 0.05])
        preferred_channel = random.choice(channels)
        subscription_status = np.random.choice(["active", "churned", "none"], p=[0.40, 0.15, 0.45])
        
        entities.append({
            "entity_id": entity_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": primary_email,
            "phone": primary_phone,
            "city": city,
            "country": country,
            "created_at": created_at,
            "churn_propensity": churn_propensity,
            "cltv_archetype": cltv_archetype,
            "preferred_channel": preferred_channel,
            "subscription_status": subscription_status,
        })
        
    df_entities = pd.DataFrame(entities)
    
    # 2. Generate Identity Links (Fragmented Touchpoints for Identity Resolution)
    # Simulates cookies, devices, CRM IDs, alternate emails that must be unified
    identity_records = []
    
    for _, ent in df_entities.iterrows():
        ent_id = ent["entity_id"]
        
        # Every entity has a primary CRM record
        crm_id = f"CRM_{ent_id[4:]}"
        identity_records.append({
            "source_system": "salesforce_crm",
            "identifier_type": "crm_id",
            "identifier_value": crm_id,
            "canonical_entity_id": ent_id,
            "confidence_score": 1.0,
            "created_at": ent["created_at"]
        })
        
        # Primary Email
        identity_records.append({
            "source_system": "marketing_cloud",
            "identifier_type": "email",
            "identifier_value": ent["email"].lower().strip(),
            "canonical_entity_id": ent_id,
            "confidence_score": 1.0,
            "created_at": ent["created_at"]
        })
        
        # Phone (Deterministic)
        identity_records.append({
            "source_system": "shopify",
            "identifier_type": "phone",
            "identifier_value": "".join(filter(str.isdigit, ent["phone"]))[-10:],
            "canonical_entity_id": ent_id,
            "confidence_score": 0.95,
            "created_at": ent["created_at"]
        })
        
        # Web cookie / Anonymous tracking cookie
        cookie_id = f"ck_{uuid.uuid4().hex[:12]}"
        identity_records.append({
            "source_system": "web_sdk",
            "identifier_type": "cookie_id",
            "identifier_value": cookie_id,
            "canonical_entity_id": ent_id,
            "confidence_score": 0.85,
            "created_at": ent["created_at"]
        })
        
        # 40% of users have a secondary mobile device ID
        if random.random() < 0.4:
            device_id = f"idfa_{uuid.uuid4().hex[:12]}"
            identity_records.append({
                "source_system": "mobile_app",
                "identifier_type": "device_id",
                "identifier_value": device_id,
                "canonical_entity_id": ent_id,
                "confidence_score": 0.90,
                "created_at": ent["created_at"] + timedelta(days=random.randint(1, 30))
            })
            
        # 15% of users have an alternate work or secondary email (probabilistic or linked via transaction)
        if random.random() < 0.15:
            alt_email = f"{ent['first_name'].lower()}{random.randint(100, 999)}@{fake.free_email_domain()}"
            identity_records.append({
                "source_system": "guest_checkout",
                "identifier_type": "email",
                "identifier_value": alt_email.lower().strip(),
                "canonical_entity_id": ent_id,
                "confidence_score": 0.80,
                "created_at": ent["created_at"] + timedelta(days=random.randint(5, 60))
            })
            
    df_identities = pd.DataFrame(identity_records)
    
    # 3. Generate Transactions / Order History
    transactions = []
    archetype_multipliers = {"low": (1, 2, 25, 60), "medium": (2, 5, 50, 150), "high": (4, 12, 100, 350), "vip": (8, 25, 200, 800)}
    
    order_counter = 1
    for _, ent in df_entities.iterrows():
        min_orders, max_orders, min_rev, max_rev = archetype_multipliers[ent["cltv_archetype"]]
        num_orders = random.randint(min_orders, max_orders) if ent["subscription_status"] != "none" or random.random() > 0.2 else (1 if random.random() > 0.3 else 0)
        
        last_date = ent["created_at"]
        for _ in range(num_orders):
            days_gap = random.randint(5, 75)
            order_date = last_date + timedelta(days=days_gap)
            if order_date > end_date:
                break
            last_date = order_date
            
            revenue = round(random.uniform(min_rev, max_rev), 2)
            margin_rate = random.uniform(0.35, 0.65)
            gross_margin = round(revenue * margin_rate, 2)
            channel = np.random.choice(channels, p=[0.25, 0.20, 0.15, 0.25, 0.05, 0.05, 0.05])
            
            transactions.append({
                "order_id": f"ORD_{order_counter:07d}",
                "entity_id": ent["entity_id"],
                "order_timestamp": order_date,
                "revenue": revenue,
                "gross_margin": gross_margin,
                "channel": channel,
                "payment_status": "completed",
                "items_count": random.randint(1, 5)
            })
            order_counter += 1
            
    df_transactions = pd.DataFrame(transactions)
    
    # 4. Generate Web & App Behavioral Events
    event_types = ["page_view", "product_view", "pricing_view", "add_to_cart", "checkout_step", "lead_form_submitted", "support_ticket"]
    events = []
    event_counter = 1
    
    for _, ent in df_entities.iterrows():
        # High-value entities have more events
        num_events = random.randint(10, 50) if ent["cltv_archetype"] in ["high", "vip"] else random.randint(3, 20)
        
        curr_time = ent["created_at"]
        for _ in range(num_events):
            curr_time = curr_time + timedelta(hours=random.randint(1, 120))
            if curr_time > end_date:
                break
            
            ev_type = np.random.choice(event_types, p=[0.40, 0.25, 0.12, 0.10, 0.06, 0.04, 0.03])
            events.append({
                "event_id": f"EVT_{event_counter:08d}",
                "entity_id": ent["entity_id"],
                "event_timestamp": curr_time,
                "event_type": ev_type,
                "page_url": f"https://mystore.com/{ev_type.replace('_', '/')}",
                "session_id": f"sess_{random.randint(1000, 9999)}",
                "device_type": random.choice(["desktop", "mobile", "tablet"])
            })
            event_counter += 1
            
    df_events = pd.DataFrame(events)
    
    # 5. Generate Campaign Touchpoints (for Multi-Touch Attribution)
    touchpoints = []
    touch_counter = 1
    campaign_names = {
        "paid_search": ["SEM_Brand_Search", "SEM_Category_HighIntent"],
        "paid_social": ["Meta_Prospecting_Video", "Meta_Retargeting_Catalog"],
        "email_marketing": ["Email_Welcome_Series", "Email_Promo_Discount", "Email_Reactivation_Winback"],
        "organic_search": ["SEO_Blog_Guide", "SEO_LandingPage"],
        "display": ["Display_BrandAwareness", "Display_DynamicRetargeting"],
        "referral": ["Affiliate_PartnerNetwork"],
        "direct": ["Direct_Navigation"]
    }
    
    # Associate campaign touchpoints prior to orders or stand-alone
    for _, ent in df_entities.iterrows():
        ent_orders = df_transactions[df_transactions["entity_id"] == ent["entity_id"]].sort_values("order_timestamp")
        
        if len(ent_orders) > 0:
            for _, ord_row in ent_orders.iterrows():
                # Generate 2 to 4 touchpoints leading up to this order
                num_touches = random.randint(1, 4)
                ord_time = ord_row["order_timestamp"]
                
                touch_times = sorted([ord_time - timedelta(days=random.randint(1, 28), hours=random.randint(1, 23)) for _ in range(num_touches)])
                
                for idx, t_time in enumerate(touch_times):
                    # Progression: Top-of-funnel channels earlier, bottom-of-funnel later
                    if idx == 0 and num_touches > 1:
                        ch = random.choice(["paid_social", "display", "organic_search"])
                    elif idx == num_touches - 1:
                        ch = random.choice(["paid_search", "email_marketing", "direct"])
                    else:
                        ch = random.choice(channels)
                        
                    camp = random.choice(campaign_names[ch])
                    is_converting = 1 if (idx == num_touches - 1) else 0
                    
                    touchpoints.append({
                        "touchpoint_id": f"TP_{touch_counter:08d}",
                        "entity_id": ent["entity_id"],
                        "event_timestamp": t_time,
                        "channel": ch,
                        "campaign_name": camp,
                        "conversion_flag": is_converting,
                        "order_id": ord_row["order_id"] if is_converting else None,
                        "attributed_revenue": ord_row["revenue"] if is_converting else 0.0
                    })
                    touch_counter += 1
        else:
            # Non-converting touches
            for _ in range(random.randint(1, 3)):
                t_time = ent["created_at"] + timedelta(days=random.randint(1, 40))
                if t_time <= end_date:
                    ch = random.choice(channels)
                    camp = random.choice(campaign_names[ch])
                    touchpoints.append({
                        "touchpoint_id": f"TP_{touch_counter:08d}",
                        "entity_id": ent["entity_id"],
                        "event_timestamp": t_time,
                        "channel": ch,
                        "campaign_name": camp,
                        "conversion_flag": 0,
                        "order_id": None,
                        "attributed_revenue": 0.0
                    })
                    touch_counter += 1
                    
    df_touchpoints = pd.DataFrame(touchpoints)
    
    # 6. Consent Preferences (GDPR / CCPA compliance for Governance)
    consents = []
    for _, ent in df_entities.iterrows():
        consents.append({
            "entity_id": ent["entity_id"],
            "marketing_opt_in": bool(random.random() > 0.15),  # 85% opt-in
            "tracking_consent": bool(random.random() > 0.10),  # 90% tracking allowed
            "third_party_sharing": bool(random.random() > 0.35), # 65% third-party sharing allowed
            "do_not_track": bool(random.random() < 0.08),      # 8% DNT active
            "updated_at": ent["created_at"]
        })
    df_consents = pd.DataFrame(consents)
    
    print(f"Data generation complete:")
    print(f" - Entities: {len(df_entities)}")
    print(f" - Identity Links: {len(df_identities)}")
    print(f" - Transactions: {len(df_transactions)}")
    print(f" - Behavioral Events: {len(df_events)}")
    print(f" - Campaign Touchpoints: {len(df_touchpoints)}")
    print(f" - Consent Records: {len(df_consents)}")
    
    return {
        "entities": df_entities,
        "identities": df_identities,
        "transactions": df_transactions,
        "events": df_events,
        "touchpoints": df_touchpoints,
        "consents": df_consents
    }

if __name__ == "__main__":
    generate_cdp_dataset()
