import requests
import json
import time

BASE_URL = "http://localhost:8000/local_fix"

test_cases = [
    {
        "name": "SQL Injection (Python)",
        "language": "python",
        "cwe": "CWE-89",
        "code": r"""import sqlite3

def get_user(username):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Vulnerable: direct string concatenation
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()"""
    },
    {
        "name": "SQL Injection (Java)",
        "language": "java",
        "cwe": "CWE-89",
        "code": r"""import java.sql.*;

public class Login {
    public boolean authenticate(String username, String password) {
        // Vulnerable: string concatenation
        String sql = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
        try {
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(sql);
            return rs.next();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return false;
    }
}"""
    },
    {
        "name": "Command Injection (Python)",
        "language": "python",
        "cwe": "CWE-78",
        "code": r"""import os

def ping_host(hostname):
    # Vulnerable: os.system with user input
    command = "ping -c 1 " + hostname
    os.system(command)"""
    }
]

def print_separator(char='-', length=60):
    print(char * length)

print(f"Starting tests against {BASE_URL}...\n")

for i, case in enumerate(test_cases, 1):
    print_separator('=')
    print(f"TEST CASE {i}: {case['name']}")
    print(f"CWE: {case['cwe']} ({case['language']})")
    print_separator()
    
    print("Original Code:")
    print(case['code'].strip())
    print_separator()

    start_time = time.time()
    try:
        response = requests.post(BASE_URL, json={
            "language": case["language"],
            "cwe": case["cwe"],
            "code": case["code"]
        })
        client_latency = (time.time() - start_time) * 1000
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            server_latency = data.get('latency_ms', 'N/A')
            
            print(f"Latency: Client={client_latency:.2f}ms, Server={server_latency}ms")
            print_separator()
            
            print("FIXED CODE:")
            print(data.get('fixed_code', 'N/A'))
            
            print("\nEXPLANATION:")
            print(data.get('explanation', 'N/A'))
            
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request Failed: {e}")
    
    print("\n")

print("Testing Complete.")
