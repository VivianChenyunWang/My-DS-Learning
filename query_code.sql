use sqlproject;
SELECT physician_id, alert_count
FROM (SELECT physician_id, COUNT(physician_id) AS alert_count,
	  RANK() OVER(ORDER BY COUNT(physician_id) DESC) AS alert_rank
      FROM alerts
      GROUP BY physician_id) AS ac
WHERE alert_rank= 1;

SELECT p1.physician_id
FROM prescriptions p1, prescriptions p2, adverse_reactions r
WHERE p1.patient_id=p2.patient_id
AND p1.physician_id=p2.physician_id
AND ((p1.drug_name=r.drug_name AND p2.drug_name=r.drug_name_2)
OR (p1.drug_name=r.drug_name_2 AND p2.drug_name=r.drug_name))
GROUP BY p1.physician_id;

SELECT physician_id, drug_count
FROM (SELECT p.physician_id, COUNT(p.physician_id) AS drug_count,
	  RANK() OVER(ORDER BY COUNT(p.physician_id) DESC) AS use_rank
      FROM prescriptions p, contracts t, companies m
      WHERE p.drug_name = t.drug
      AND m.id = t.company_id
	  AND m.name = 'DRUGXO'
      GROUP BY p.physician_id) AS x
WHERE use_rank = 1;

SELECT t.drug, CAST(price/quantity AS DECIMAL(10,2)) AS price, 
	   CAST(AVG(price/quantity) OVER(PARTITION BY (drug)) AS DECIMAL(10,2)) AS avg_price
FROM companies m, contracts t
WHERE m.id=t.company_id
AND m.name= 'PHARMASEE';

SELECT p.drug_name, c.name AS phar_name,
	   CAST((f.cost/p.quantity-t.price/t.quantity)/(t.price/t.quantity)*100 AS DECIMAL(10,2)) AS percent_markup
FROM prescriptions p, pharmacies c, pharmacy_fills f, contracts t
WHERE p.id=f.prescription_id
AND f.pharmacy_id=t.pharmacy_id
AND c.id=f.pharmacy_id
AND p.drug_name=t.drug;

SELECT p.drug_name, AVG(DATEDIFF(f.date, p.date)) AS avg_daydiff
FROM prescriptions p, pharmacy_fills f
WHERE p.id=f.prescription_id
GROUP BY p.drug_name;

SELECT c.id AS pharmacy_id, p.drug_name
FROM pharmacies c
CROSS JOIN (SELECT DISTINCT drug_name FROM prescriptions) p
LEFT JOIN  (SELECT f.pharmacy_id, p.drug_name 
            FROM pharmacy_fills f, prescriptions p
			WHERE p.id = f.prescription_id) d
ON d.pharmacy_id = c.id AND d.drug_name = p.drug_name
WHERE d.pharmacy_id IS NULL;

