(define (problem kitchen_problem)
  (:domain kitchen)
  (:objects 
    stove button pot bread serving table - object
  )
  (:init 
    (free-gripper)
    (on stove table)
    (on button table)
    (on pot table)
    (on bread table)
    (on serving table)
    (on table table)
  )
  (:goal 
    (and (not (stove_on))
         (cooked bread)
         (on pot serving)
         (on bread pot)))
)