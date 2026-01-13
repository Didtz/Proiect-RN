-- House Plants Database Schema
-- Complete database structure for plant management system

DROP DATABASE IF EXISTS house_plants_db;
CREATE DATABASE house_plants_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE house_plants_db;

-- Table: plants
-- Main table storing all plant information
CREATE TABLE plants (
    id INT PRIMARY KEY,
    latin_name VARCHAR(255) NOT NULL,
    family VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    origin VARCHAR(255),
    climate VARCHAR(100),
    temp_max_celsius DECIMAL(5,2),
    temp_max_fahrenheit DECIMAL(5,2),
    temp_min_celsius DECIMAL(5,2),
    temp_min_fahrenheit DECIMAL(5,2),
    ideal_light TEXT,
    tolerated_light TEXT,
    watering TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_family (family),
    INDEX idx_latin_name (latin_name)
) ENGINE=InnoDB;

-- Table: common_names
-- Stores multiple common names per plant (one-to-many relationship)
CREATE TABLE common_names (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plant_id INT NOT NULL,
    common_name VARCHAR(255) NOT NULL,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    INDEX idx_plant_id (plant_id),
    INDEX idx_common_name (common_name)
) ENGINE=InnoDB;

-- Table: insects
-- Stores insect/pest information that can affect plants
CREATE TABLE insects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    treatment TEXT
) ENGINE=InnoDB;

-- Table: plant_insects
-- Junction table for many-to-many relationship between plants and insects
CREATE TABLE plant_insects (
    plant_id INT NOT NULL,
    insect_id INT NOT NULL,
    severity ENUM('low', 'medium', 'high') DEFAULT 'medium',
    PRIMARY KEY (plant_id, insect_id),
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    FOREIGN KEY (insect_id) REFERENCES insects(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Table: diseases
-- Stores disease information that can affect plants
CREATE TABLE diseases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    treatment TEXT
) ENGINE=InnoDB;

-- Table: plant_diseases
-- Junction table for many-to-many relationship between plants and diseases
CREATE TABLE plant_diseases (
    plant_id INT NOT NULL,
    disease_id INT NOT NULL,
    severity ENUM('low', 'medium', 'high') DEFAULT 'medium',
    PRIMARY KEY (plant_id, disease_id),
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    FOREIGN KEY (disease_id) REFERENCES diseases(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Table: plant_uses
-- Stores different uses/purposes for plants
CREATE TABLE plant_uses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plant_id INT NOT NULL,
    use_type VARCHAR(100) NOT NULL,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    INDEX idx_plant_id (plant_id),
    INDEX idx_use_type (use_type)
) ENGINE=InnoDB;

-- Table: care_schedules
-- Track care schedules for individual plants
CREATE TABLE care_schedules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plant_id INT NOT NULL,
    last_watered DATE,
    last_fertilized DATE,
    last_pruned DATE,
    next_watering_due DATE,
    notes TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    INDEX idx_plant_id (plant_id),
    INDEX idx_next_watering (next_watering_due)
) ENGINE=InnoDB;

-- Table: plant_images
-- Store paths to plant images for the neural network training
CREATE TABLE plant_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plant_id INT NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    image_type ENUM('training', 'validation', 'test') DEFAULT 'training',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    INDEX idx_plant_id (plant_id),
    INDEX idx_image_type (image_type)
) ENGINE=InnoDB;

-- Table: user_plants
-- Track plants owned by users (if you add user functionality)
CREATE TABLE user_plants (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plant_id INT NOT NULL,
    user_id INT,
    nickname VARCHAR(255),
    purchase_date DATE,
    location VARCHAR(255),
    notes TEXT,
    is_healthy BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE,
    INDEX idx_plant_id (plant_id),
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB;

-- Table: categories
-- Store all unique plant categories
CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
) ENGINE=InnoDB;

-- Table: families
-- Store all unique plant families
CREATE TABLE families (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
) ENGINE=InnoDB;

-- Views for easier querying
CREATE VIEW plant_full_info AS
SELECT 
    p.*,
    GROUP_CONCAT(DISTINCT cn.common_name SEPARATOR ', ') as common_names,
    GROUP_CONCAT(DISTINCT pu.use_type SEPARATOR ', ') as uses
FROM plants p
LEFT JOIN common_names cn ON p.id = cn.plant_id
LEFT JOIN plant_uses pu ON p.id = pu.plant_id
GROUP BY p.id;

CREATE VIEW plants_with_pests AS
SELECT 
    p.id,
    p.latin_name,
    GROUP_CONCAT(DISTINCT i.name SEPARATOR ', ') as insects,
    GROUP_CONCAT(DISTINCT d.name SEPARATOR ', ') as diseases
FROM plants p
LEFT JOIN plant_insects pi ON p.id = pi.plant_id
LEFT JOIN insects i ON pi.insect_id = i.id
LEFT JOIN plant_diseases pd ON p.id = pd.plant_id
LEFT JOIN diseases d ON pd.disease_id = d.id
GROUP BY p.id, p.latin_name;