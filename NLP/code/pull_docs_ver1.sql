select distinct id, publishdate, news
from usnewspaper
where news ilike '%overdose%' or
      news ilike '%prescription drug%' or
      news ilike '%fentanyl%' or
      news ilike '%opioid%' or
      news ilike '%prescription drug%' or
      news ilike '%addiction%' or
      news ilike '%drug law%' or
      news ilike '%oxycodon%' or
      news ilike '%heroin%' or
      news ilike '%drug%treatment%' or
      news ilike '%drug%enforcement%' or
      news ilike '%drug%policy%' or
      news ilike '%drug%seizure%' or
      news ilike '%drug%arrest%' or
      news ilike '%drug%crime%' or
      news ilike '%drug%market%' or
      news ilike '%drug%cartel%' or
      news ilike '%drug%trend%' or
      news ilike '%recreation%drug%' or
      news ilike '%drug%depend%' or
      news ilike '%drug%disorder%' or
      news ilike '%drug%prescription%' or
      news ilike '%drug%pain%';