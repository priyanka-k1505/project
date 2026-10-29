drop database if exists kidney;
create database kidney;
use kidney;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(225),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255)   -- stores a salted hash, not plain text
);
