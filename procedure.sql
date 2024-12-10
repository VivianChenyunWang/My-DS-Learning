USE sqlproject;

DELIMITER //

CREATE PROCEDURE GetPhysicianDetails (
    IN physician_ssn VARCHAR(128), 
    OUT physician_specialty VARCHAR(128),
    OUT experience_years INT
)
BEGIN
    SELECT primary_specialty, experience_years 
    INTO physician_specialty, experience_years
    FROM physicians
    WHERE physicians.SSN = physician_ssn;
END //

DELIMITER ;
