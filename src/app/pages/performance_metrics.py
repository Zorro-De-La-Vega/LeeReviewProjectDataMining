#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Metrics Dashboard Module

This module provides a performance monitoring dashboard for the application,
showing execution times, memory usage, and optimization opportunities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import gc

def show_performance_metrics_page(data_loader=None, agent=None):
    """
    Display the performance metrics dashboard page.
    
    This page shows performance metrics for the application,
    helping diagnose bottlenecks and optimize performance.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: HousingImpactAgent instance for proactive insights
    """
    st.header("Performance Metrics Dashboard")
    
    st.markdown("""
        This dashboard provides insights into the application's performance metrics,
        helping identify bottlenecks and optimization opportunities.
    """)
    
    # System-level metrics
    with st.container():
        st.subheader("System Resources")
        
        # Get current process info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Memory usage in MB
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_usage_mb:.1f} MB")
        
        with col2:
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        
        with col3:
            # Uptime
            if 'start_time' not in st.session_state:
                st.session_state['start_time'] = time.time()
            uptime_seconds = time.time() - st.session_state['start_time']
            uptime_formatted = str(timedelta(seconds=int(uptime_seconds)))
            st.metric("App Uptime", uptime_formatted)
        
        with col4:
            # Number of active sessions/connections
            active_sessions = 1  # Default to 1 (current session)
            st.metric("Active Sessions", active_sessions)
    
    # Performance metrics from session state
    st.subheader("Page Performance Metrics")
    
    if 'page_metrics' not in st.session_state:
        st.session_state['page_metrics'] = {}
    
    # Create metrics for the current page view if not already done
    current_page = st.session_state.get('current_page', 'Dashboard')
    page_load_time = time.time() - st.session_state.get('page_load_start_time', time.time())
    
    if current_page not in st.session_state['page_metrics']:
        st.session_state['page_metrics'][current_page] = {
            'load_times': [],
            'render_count': 0
        }
    
    st.session_state['page_metrics'][current_page]['load_times'].append(page_load_time)
    st.session_state['page_metrics'][current_page]['render_count'] += 1
    
    # Keep only the last 10 load times
    if len(st.session_state['page_metrics'][current_page]['load_times']) > 10:
        st.session_state['page_metrics'][current_page]['load_times'] = st.session_state['page_metrics'][current_page]['load_times'][-10:]
    
    # Convert metrics to DataFrame
    metrics_data = []
    
    for page, data in st.session_state['page_metrics'].items():
        avg_load_time = sum(data['load_times']) / len(data['load_times']) if data['load_times'] else 0
        metrics_data.append({
            'Page': page,
            'Average Load Time (s)': round(avg_load_time, 3),
            'Last Load Time (s)': round(data['load_times'][-1] if data['load_times'] else 0, 3),
            'Render Count': data['render_count']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Sort by load time
    metrics_df = metrics_df.sort_values('Average Load Time (s)', ascending=False)
    
    # Show metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Visualization of load times
    if not metrics_df.empty:
        fig = px.bar(
            metrics_df,
            x='Page',
            y='Average Load Time (s)',
            color='Average Load Time (s)',
            color_continuous_scale='RdYlGn_r',  # Red for slow, green for fast
            title='Average Page Load Times'
        )
        
        # Add threshold line for acceptable performance
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="1s Threshold", 
                     annotation_position="top right")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Dashboard performance metrics
    if 'dashboard_performance' in st.session_state:
        st.subheader("Dashboard Section Performance")
        
        dashboard_metrics = []
        for section, data in st.session_state['dashboard_performance'].items():
            dashboard_metrics.append({
                'Section': section,
                'Load Time (s)': round(data['load_time'], 3),
                'Renders': data['renders']
            })
        
        dashboard_df = pd.DataFrame(dashboard_metrics)
        dashboard_df = dashboard_df.sort_values('Load Time (s)', ascending=False)
        
        st.dataframe(dashboard_df, use_container_width=True)
    
    # Visualization performance metrics
    if 'vis_performance' in st.session_state:
        st.subheader("Visualization Performance")
        
        vis_metrics = []
        for func_name, stats in st.session_state['vis_performance'].items():
            vis_metrics.append({
                'Visualization Function': func_name,
                'Last Execution Time (s)': round(stats['last_execution_time'], 3),
                'Call Count': stats['call_count']
            })
        
        vis_df = pd.DataFrame(vis_metrics)
        vis_df = vis_df.sort_values('Last Execution Time (s)', ascending=False)
        
        st.dataframe(vis_df, use_container_width=True)
        
        # Visualization of execution times
        if not vis_df.empty:
            fig = px.bar(
                vis_df,
                x='Visualization Function',
                y='Last Execution Time (s)',
                color='Last Execution Time (s)',
                color_continuous_scale='RdYlGn_r',
                title='Visualization Function Execution Times'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance optimization actions
    st.subheader("Performance Optimization Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Session State Cache"):
            # Keep only essential session state items
            keep_keys = ['start_time', 'current_page', 'page_load_start_time']
            for key in list(st.session_state.keys()):
                if key not in keep_keys:
                    del st.session_state[key]
            st.success("Session state cache cleared!")
            st.experimental_rerun()
    
    with col2:
        if st.button("Run Garbage Collection"):
            # Force garbage collection
            collected = gc.collect()
            st.success(f"Garbage collection completed! Collected {collected} objects.")
    
    # Optimization recommendations
    st.subheader("Performance Optimization Recommendations")
    
    # Identify slow pages (over 1 second)
    slow_pages = metrics_df[metrics_df['Average Load Time (s)'] > 1]['Page'].tolist()
    
    recommendations = []
    
    if slow_pages:
        recommendations.append(f"The following pages have load times over 1 second: {', '.join(slow_pages)}")
        recommendations.append("Consider the following optimizations for these pages:")
        recommendations.append("- Implement more aggressive data caching with `st.cache_data`")
        recommendations.append("- Use lazy loading for heavy components")
        recommendations.append("- Sample large datasets to improve rendering speed")
        recommendations.append("- Consider using Plotly's WebGL rendering for large plots")
    
    # Check memory usage
    if memory_usage_mb > 500:  # Over 500 MB
        recommendations.append("Memory usage is high. Consider:")
        recommendations.append("- Optimize dataframe memory usage with downcasting")
        recommendations.append("- Use garbage collection more aggressively")
        recommendations.append("- Limit the size of caches")
    
    # Check for visualization performance issues
    if 'vis_performance' in st.session_state:
        slow_vis = [func for func, stats in st.session_state['vis_performance'].items() 
                    if stats['last_execution_time'] > 1]
        if slow_vis:
            recommendations.append(f"Slow visualization functions: {', '.join(slow_vis)}")
            recommendations.append("- Consider sampling data for these visualizations")
            recommendations.append("- Use aggressive caching with appropriate TTLs")
            recommendations.append("- Simplify visualizations or use more efficient libraries")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("No performance issues detected. The application is running optimally!")
    
    # Manual performance testing
    with st.expander("Run Manual Performance Test", expanded=False):
        st.markdown("Test the performance of specific components:")
        
        test_options = [
            "Data Loading",
            "Dashboard Rendering",
            "Advanced Analysis",
            "Visualization"
        ]
        
        test_selection = st.selectbox("Select component to test:", test_options)
        
        if st.button("Run Test"):
            with st.spinner(f"Testing {test_selection} performance..."):
                start_time = time.time()
                
                # Perform the selected test
                if test_selection == "Data Loading":
                    # Test data loading performance
                    if data_loader:
                        data_loader.get_airbnb_data()
                        data_loader.get_census_data()
                        data_loader.get_combined_data()
                
                elif test_selection == "Dashboard Rendering":
                    # Test dashboard rendering (simplified)
                    if data_loader:
                        combined_data = data_loader.get_combined_data()
                        if combined_data is not None and not combined_data.empty:
                            # Sample calculation to simulate dashboard rendering
                            combined_data.describe()
                
                elif test_selection == "Advanced Analysis":
                    # Test analysis calculations
                    if data_loader:
                        combined_data = data_loader.get_combined_data()
                        if combined_data is not None and not combined_data.empty:
                            # Sample calculation to simulate advanced analysis
                            for col in combined_data.select_dtypes(include=['number']).columns:
                                combined_data[col].mean()
                                combined_data[col].std()
                
                elif test_selection == "Visualization":
                    # Test visualization performance
                    if data_loader:
                        combined_data = data_loader.get_combined_data()
                        if combined_data is not None and not combined_data.empty:
                            # Create a sample visualization
                            if 'neighborhood' in combined_data.columns and 'airbnb_count' in combined_data.columns:
                                fig = px.bar(combined_data, x='neighborhood', y='airbnb_count')
                
                elapsed_time = time.time() - start_time
                
                # Show test results
                st.success(f"Test completed in {elapsed_time:.3f} seconds")
                
                # Store test result
                if 'performance_tests' not in st.session_state:
                    st.session_state['performance_tests'] = []
                
                st.session_state['performance_tests'].append({
                    'component': test_selection,
                    'execution_time': elapsed_time,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Show previous test results
        if 'performance_tests' in st.session_state and st.session_state['performance_tests']:
            st.markdown("### Previous Test Results")
            
            test_results_df = pd.DataFrame(st.session_state['performance_tests'])
            st.dataframe(test_results_df, use_container_width=True)
            
            # Visualize test results
            fig = px.line(
                test_results_df,
                x='timestamp',
                y='execution_time',
                color='component',
                title='Performance Test Results Over Time'
            )
            
            st.plotly_chart(fig, use_container_width=True)
