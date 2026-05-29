"""
Example of using the EV-QA-Framework configuration system

Demonstrates:
1. Creating custom configurations
2. Saving/loading configurations
3. Using different configurations for different battery types
"""

import asyncio
from ev_qa_framework import (
    EVQAFramework, 
    FrameworkConfig, 
    SafetyThresholds, 
    MLConfig
)


def example_1_default_config():
    """Example 1: Using the default configuration"""
    print("=" * 60)
    print("Example 1: Default Configuration")
    print("=" * 60)
    
    qa = EVQAFramework("Default-QA")
    
    print(f"Max temperature: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Voltage range: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print(f"ML contamination: {qa.config.ml_config.contamination}")
    print()


def example_2_tesla_config():
    """Example 2: Strict configuration for Tesla"""
    print("=" * 60)
    print("Example 2: Tesla Configuration (strict thresholds)")
    print("=" * 60)
    
    # Creating strict thresholds for Tesla
    tesla_thresholds = SafetyThresholds(
        max_temperature=55.0,      # Tesla is more conservative
        min_temperature=-20.0,      # Operation in moderate climate
        max_temperature_jump=3.0,   # Very sensitive to jumps
        min_voltage=250.0,          # Narrower range
        max_voltage=450.0,
        min_soc=20.0,              # Warning at SOC < 20%
        critical_soh=80.0,         # Higher health threshold
        max_current=600.0
    )
    
    # ML configuration with high accuracy
    tesla_ml = MLConfig(
        contamination=0.05,        # Expect fewer anomalies
        n_estimators=250,          # More trees = more accurate
        critical_score_threshold=-0.9,  # Stricter threshold
        warning_score_threshold=-0.6
    )
    
    config = FrameworkConfig(
        safety_thresholds=tesla_thresholds,
        ml_config=tesla_ml,
        default_vin="5YJ3E1EA8KF000001"  # Tesla VIN
    )
    
    # Save configuration
    config.save_to_file("tesla_custom.json")
    print("✅ Tesla configuration saved to tesla_custom.json")
    
    qa = EVQAFramework("Tesla-QA", config=config)
    print(f"Max temperature: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Voltage range: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print(f"ML contamination: {qa.config.ml_config.contamination}")
    print()


def example_3_nissan_leaf_config():
    """Example 3: Configuration for Nissan Leaf (softer thresholds)"""
    print("=" * 60)
    print("Example 3: Nissan Leaf Configuration")
    print("=" * 60)
    
    # Nissan Leaf has different characteristics
    leaf_thresholds = SafetyThresholds(
        max_temperature=65.0,      # Less strict threshold
        min_temperature=-30.0,
        max_temperature_jump=7.0,  # More tolerant of jumps
        min_voltage=300.0,         # Different voltage range
        max_voltage=400.0,
        min_soc=10.0,
        critical_soh=65.0,
        max_current=400.0
    )
    
    leaf_ml = MLConfig(
        contamination=0.15,        # More anomalies expected
        n_estimators=150
    )
    
    config = FrameworkConfig(
        safety_thresholds=leaf_thresholds,
        ml_config=leaf_ml,
        default_vin="1N4AZ0CP0FC000001"  # Nissan Leaf VIN
    )
    
    config.save_to_file("nissan_leaf.json")
    print("✅ Nissan Leaf configuration saved to nissan_leaf.json")
    
    qa = EVQAFramework("Leaf-QA", config=config)
    print(f"Max temperature: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Voltage range: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print()


async def example_4_testing_with_config():
    """Example 4: Testing with a custom configuration"""
    print("=" * 60)
    print("Example 4: Testing with a custom configuration")
    print("=" * 60)
    
    # Load saved configuration
    config = FrameworkConfig.load_from_file("tesla_custom.json")
    qa = EVQAFramework("Test-QA", config=config)
    
    # Test data
    test_data = [
        {'voltage': 350.0, 'current': 100, 'temperature': 30, 'soc': 80, 'soh': 95},
        {'voltage': 355.0, 'current': 95, 'temperature': 31, 'soc': 78, 'soh': 95},
        {'voltage': 360.0, 'current': 90, 'temperature': 32, 'soc': 76, 'soh': 95},
        {'voltage': 365.0, 'current': 85, 'temperature': 33, 'soc': 74, 'soh': 95},
        {'voltage': 370.0, 'current': 80, 'temperature': 56, 'soc': 72, 'soh': 95},  # Temperature > 55°C!
    ]
    
    results = await qa.run_test_suite(test_data)
    
    print(f"📊 Test Results:")
    print(f"   Total tests: {results['total_tests']}")
    print(f"   Passed: {results['passed']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Anomalies: {len(results['anomalies'])}")
    
    if results['ml_analysis']:
        print(f"   ML severity: {results['ml_analysis']['severity']}")
    
    print()


def example_5_comparison():
    """Example 5: Comparing configurations"""
    print("=" * 60)
    print("Example 5: Comparing Configurations")
    print("=" * 60)
    
    # Load different configurations
    tesla_cfg = FrameworkConfig.load_from_file("tesla_custom.json")
    leaf_cfg = FrameworkConfig.load_from_file("nissan_leaf.json")
    
    print("Tesla vs Nissan Leaf Comparison:")
    print("-" * 60)
    print(f"{'Parameter':<30} {'Tesla':<15} {'Leaf':<15}")
    print("-" * 60)
    print(f"{'Max temperature':<30} {tesla_cfg.safety_thresholds.max_temperature:<15} {leaf_cfg.safety_thresholds.max_temperature:<15}")
    print(f"{'Max voltage':<30} {tesla_cfg.safety_thresholds.max_voltage:<15} {leaf_cfg.safety_thresholds.max_voltage:<15}")
    print(f"{'Temperature jump':<30} {tesla_cfg.safety_thresholds.max_temperature_jump:<15} {leaf_cfg.safety_thresholds.max_temperature_jump:<15}")
    print(f"{'ML contamination':<30} {tesla_cfg.ml_config.contamination:<15} {leaf_cfg.ml_config.contamination:<15}")
    print()


def main():
    """Main function - run all examples"""
    print("\n🚗 EV-QA-Framework: Configuration Examples\n")
    
    # Run examples
    example_1_default_config()
    example_2_tesla_config()
    example_3_nissan_leaf_config()
    
    # Async example
    asyncio.run(example_4_testing_with_config())
    
    example_5_comparison()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
