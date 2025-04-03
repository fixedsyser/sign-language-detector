# Sign Language Alphabet Recognition

## Project Description

This project aims to develop a sign language alphabet recognition system using computer vision and machine learning techniques. Sign language is a vital communication method for many deaf or mute individuals, and creating accessible technology to bridge communication gaps is an important area of research.

The system works by leveraging hand pose estimation to detect the positions of hand joints, creating a skeleton-like representation of the hand. These joint positions serve as features for a machine learning classifier that identifies which letter of the American Sign Language (ASL) alphabet is being shown.

## Hand Pose Estimation
The project uses MediaPipe's hand landmark detection system to track 21 key points on the hand in real-time. These landmarks provide precise information about finger positions and hand orientation, which are crucial for distinguishing between different sign language gestures.
