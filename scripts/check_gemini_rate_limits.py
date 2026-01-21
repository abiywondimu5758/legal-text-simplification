# check_gemini_rate_limits.py
import google.generativeai as genai
import time
import json

API_KEY = "REPLACE WITH YOUR API KEY HERE"  # Replace with your key
genai.configure(api_key=API_KEY)

# Test Gemini 3 Flash Preview (the one we'll actually use)
MODEL_NAME = 'gemini-3-flash-preview'
model = genai.GenerativeModel(MODEL_NAME)

print("=" * 80)
print(f"ACCURATE RATE LIMIT TEST - {MODEL_NAME}")
print("=" * 80)
print("\nBased on official docs: Free tier = 5-15 RPM (varies)")
print("This test will find YOUR exact limit.\n")

# Test strategy: Start conservative, find the breaking point
def test_rate_limit(requests_per_minute_target):
    """Test if we can sustain a specific RPM"""
    interval = 60.0 / requests_per_minute_target  # Seconds between requests
    test_requests = min(requests_per_minute_target, 20)  # Test up to 20 requests
    
    print(f"\nTesting {requests_per_minute_target} RPM ({interval:.2f}s interval)...")
    
    success_count = 0
    errors = []
    start_time = time.time()
    
    for i in range(test_requests):
        try:
            response = model.generate_content("Say 'test' in Amharic")
            success_count += 1
            print(f"  [{i+1}/{test_requests}] ✓", end='\r', flush=True)
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            if "429" in error_msg or "quota" in error_msg.lower():
                elapsed = time.time() - start_time
                actual_rpm = (success_count / elapsed * 60) if elapsed > 0 else 0
                print(f"\n  ❌ Rate limit hit at request {i+1}")
                print(f"     Successful: {success_count}/{i+1}")
                print(f"     Actual RPM before limit: {actual_rpm:.1f}")
                return False, success_count, actual_rpm
            else:
                print(f"\n  ⚠️  Other error: {error_msg[:50]}")
                return False, success_count, 0
        
        if i < test_requests - 1:
            time.sleep(interval)
    
    elapsed = time.time() - start_time
    actual_rpm = (success_count / elapsed * 60) if elapsed > 0 else 0
    print(f"\n  ✅ Sustained {success_count} requests")
    print(f"     Actual RPM: {actual_rpm:.1f}")
    return True, success_count, actual_rpm

# Test different RPM values to find the limit
print("=" * 80)
print("FINDING YOUR EXACT RATE LIMIT")
print("=" * 80)

# Test from low to high
test_rpms = [5, 10, 15, 20]  # Test common free tier values
max_sustainable_rpm = 0

for i, rpm in enumerate(test_rpms):
    can_sustain, success_count, actual_rpm = test_rate_limit(rpm)
    
    if can_sustain:
        max_sustainable_rpm = rpm
        print(f"✅ {rpm} RPM: SUSTAINABLE")
        
        # Wait 60+ seconds before next test to reset rate limit window
        # (Only if there's a next test to run)
        if i < len(test_rpms) - 1:
            next_rpm = test_rpms[i + 1]
            print(f"\n⏳ Waiting 65 seconds before testing {next_rpm} RPM...")
            print("   (This resets the rate limit window)")
            for remaining in range(65, 0, -5):
                print(f"   {remaining} seconds remaining...", end='\r', flush=True)
                time.sleep(5)
            print("\n   ✅ Rate limit window reset, starting next test...\n")
    else:
        print(f"❌ {rpm} RPM: NOT SUSTAINABLE")
        break

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

if max_sustainable_rpm > 0:
    print(f"✅ Your sustainable rate limit: {max_sustainable_rpm} requests/minute")
    print(f"\nFor 800 pairs:")
    time_per_pair = 60.0 / max_sustainable_rpm
    total_time_minutes = 800 * time_per_pair / 60
    print(f"  Time per pair: {time_per_pair:.1f} seconds")
    print(f"  Total time (single key): {total_time_minutes:.1f} minutes")
    print(f"  Total time (8 keys parallel): {total_time_minutes / 8:.1f} minutes")
else:
    print("❌ Could not determine sustainable rate limit")
    print("   Your limit may be lower than 5 RPM, or quota is exhausted")

print("\n" + "=" * 80)
print("OFFICIAL DOCUMENTATION")
print("=" * 80)
print("\nAccording to Google's official docs:")
print("  - Free tier: 5-15 RPM (varies by project)")
print("  - Requests per day: 1,000 (for preview models)")
print("  - Limits are PER PROJECT, not per API key")
print("\n⚠️  Important: If you use multiple keys from the same Google account,")
print("   they may share the same project quota!")

print("\n" + "=" * 80)
print("RECOMMENDATION FOR PARALLEL PROCESSING")
print("=" * 80)
if max_sustainable_rpm > 0:
    safe_rpm = max_sustainable_rpm * 0.9  # Use 90% to be safe
    safe_interval = 60.0 / safe_rpm
    print(f"\nUse interval: {safe_interval:.2f} seconds between requests")
    print(f"This ensures you stay under {max_sustainable_rpm} RPM per key")
else:
    print("\n⚠️  Use conservative 4.1 seconds (15 RPM) as default")
    print("   Adjust based on actual test results")