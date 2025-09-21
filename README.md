# Vanguard– AI-Powered Autonomous Vehicle

# Overview
Vanguard is an AI-driven autonomous robot built using ESP32 and machine learning techniques for real-time navigation and situational awareness in hazardous or disaster-prone areas. It intelligently detects key scenarios and dynamically follows optimal paths using graph-based algorithms.

---

# Key Features

-  **Real-Time Image Classification**  
  Detects 5 scene types – *Fire*, *Military Vehicles*, *Humanitarian Aid*, *Destroyed Buildings*, and *Combat* using a custom-trained model.

-  **Autonomous Navigation**  
  Uses ArUco markers and a line-following mechanism for physical path traversal on a predefined map.

- 🗺 **Shortest Path Optimization**  
  Applies **Dijkstra’s Algorithm** for efficient and intelligent movement between detected zones.

- **Trigger-based Actions**  
  Stops at relevant detection points and activates a buzzer signal for alerts.

-  **Modular Design**  
  Clean separation between image processing, path planning, and hardware control modules.

---

## Technologies Used

- **Hardware**: ESP32, Sensors, Motors, Buzzer  
- **Computer Vision**: OpenCV, ArUco Markers  
- **Machine Learning**: Custom CNN (Python)  
- **Algorithms**: Dijkstra’s Algorithm, Graph Theory  
- **Languages**: Python, C++  
- **Tools**: Arduino IDE, Jupyter Notebook, Git

---

## 🛠️ How It Works

1. **Environment Setup**  
   - A custom map is created using ArUco markers to guide the robot via line-following.

2. **Image Classification**  
   - OpenCV captures frames in real-time and resizes them for performance.  
   - A trained CNN model classifies each frame into one of five classes.

3. **Dynamic Navigation**  
   - Based on classification, the robot travels to pre-mapped coordinates.  
   - Dijkstra’s algorithm computes the shortest path between stops.

4. **Event Handling**  
   - Upon reaching a detection zone, the robot halts and activates a buzzer.  
   - After visiting all zones, it returns to the starting point.

---



---


